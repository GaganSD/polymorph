//! Shared helpers for CSV→log-line and JSON-array→log-line corpus adapters.
//!
//! Ported from the Python `polymorph_lamr.distill.adapters._common`, including
//! the streaming JSON-array element parser (a hand-rolled state machine that
//! yields each top-level element of a named array without loading the whole
//! document structure, and skips elements that fail to parse).

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// Replace embedded newlines/carriage-returns with spaces; `None` -> "".
pub fn sanitize_field(value: Option<&str>) -> String {
    match value {
        None => String::new(),
        Some(v) => v.replace('\n', " ").replace('\r', " "),
    }
}

/// Collapse all runs of whitespace to single spaces and trim.
pub fn collapse_whitespace(line: &str) -> String {
    line.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// True iff every required column is present in `row`.
pub fn has_required_columns(row: &HashMap<String, String>, columns: &[&str]) -> bool {
    columns.iter().all(|c| row.contains_key(*c))
}

/// Sanitized field value, or empty string if the column is absent.
pub fn row_field(row: &HashMap<String, String>, column: &str) -> String {
    match row.get(column) {
        None => String::new(),
        Some(v) => sanitize_field(Some(v)),
    }
}

/// Stream `csv_path` to `out_path`, one rendered line per row. Returns
/// `(written, skipped)`. `render_row` returns `None` to skip a row.
pub fn stream_csv_to_txt<F>(
    csv_path: &Path,
    out_path: &Path,
    render_row: &mut F,
    required_columns: &[&str],
) -> Result<(usize, usize)>
where
    F: FnMut(&HashMap<String, String>) -> Option<String>,
{
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut written = 0;
    let mut skipped = 0;
    let mut rdr = csv::ReaderBuilder::new().flexible(true).from_path(csv_path)?;
    let headers = rdr.headers()?.clone();
    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    for record in rdr.records() {
        let record = record?;
        // Mirror Python csv.DictReader: every header key is present even on a short
        // row (missing fields default to "" — the sanitized form of DictReader's
        // restval=None), so `has_required_columns` / `row_field` match it exactly.
        let mut row: HashMap<String, String> = HashMap::new();
        for (i, h) in headers.iter().enumerate() {
            row.insert(h.to_string(), record.get(i).unwrap_or("").to_string());
        }
        if !has_required_columns(&row, required_columns) {
            skipped += 1;
            continue;
        }
        match render_row(&row) {
            None => skipped += 1,
            Some(line) => {
                out.write_all(line.as_bytes())?;
                out.write_all(b"\n")?;
                written += 1;
            }
        }
    }
    out.flush()?;
    Ok((written, skipped))
}

static ARRAY_KEY_CACHE: Lazy<std::sync::Mutex<HashMap<String, Regex>>> =
    Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

fn array_key_pattern(array_key: &str) -> Regex {
    let mut cache = ARRAY_KEY_CACHE.lock().unwrap();
    cache
        .entry(array_key.to_string())
        .or_insert_with(|| {
            Regex::new(&format!(r#""{}"\s*:\s*\["#, regex::escape(array_key))).unwrap()
        })
        .clone()
}

/// Yield each top-level element of a JSON array keyed by `array_key`. `None`
/// entries are elements that failed to parse (skipped by callers). Faithful port
/// of the Python char-state-machine, including its handling of strings, nesting,
/// and escapes.
pub fn iter_json_array_elements(content: &str, array_key: &str) -> Vec<Option<Value>> {
    let pat = array_key_pattern(array_key);
    let start = match pat.find(content) {
        Some(m) => m.end(),
        None => return Vec::new(),
    };
    let chars: Vec<char> = content[start..].chars().collect();

    let mut out: Vec<Option<Value>> = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    let mut collecting = false;
    let mut element_depth: i64 = 0;
    let mut element_parts = String::new();

    let emit = |parts: &str, out: &mut Vec<Option<Value>>| {
        match serde_json::from_str::<Value>(parts) {
            Ok(v) => out.push(Some(v)),
            Err(_) => out.push(None),
        }
    };

    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if !collecting {
            if matches!(ch, ' ' | '\t' | '\r' | '\n' | ',') {
                i += 1;
                continue;
            }
            if ch == ']' {
                return out;
            }
            collecting = true;
            element_parts.clear();
            element_parts.push(ch);
            if ch == '"' {
                in_string = true;
                element_depth = 0;
            } else if ch == '[' || ch == '{' {
                element_depth = 1;
            } else {
                element_depth = 0;
            }
            i += 1;
            if element_depth == 0 && ch != '"' && ch != '[' && ch != '{' {
                emit(&element_parts, &mut out);
                collecting = false;
                element_parts.clear();
            }
            continue;
        }

        element_parts.push(ch);

        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
                if element_depth == 0 {
                    emit(&element_parts, &mut out);
                    collecting = false;
                    element_parts.clear();
                }
            }
            i += 1;
            continue;
        }

        if ch == '"' {
            in_string = true;
            i += 1;
            continue;
        }
        if ch == '[' || ch == '{' {
            element_depth += 1;
            i += 1;
            continue;
        }
        if ch == ']' || ch == '}' {
            element_depth -= 1;
            if element_depth == 0 {
                emit(&element_parts, &mut out);
                collecting = false;
                element_parts.clear();
            }
            i += 1;
            continue;
        }
        i += 1;
    }
    out
}

/// Stream elements from a large JSON object's array field to log lines. Returns
/// `(written, skipped)`. `render_item` returns `None` to skip an element.
pub fn stream_json_array_to_txt<F>(
    json_path: &Path,
    out_path: &Path,
    array_key: &str,
    render_item: &mut F,
) -> Result<(usize, usize)>
where
    F: FnMut(&Value) -> Option<String>,
{
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = std::fs::read_to_string(json_path)?;
    let mut written = 0;
    let mut skipped = 0;
    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    for element in iter_json_array_elements(&content, array_key) {
        match element {
            None => skipped += 1,
            Some(v) => match render_item(&v) {
                None => skipped += 1,
                Some(line) => {
                    out.write_all(line.as_bytes())?;
                    out.write_all(b"\n")?;
                    written += 1;
                }
            },
        }
    }
    out.flush()?;
    Ok((written, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn sanitize_field_replaces_newlines() {
        assert_eq!(sanitize_field(Some("a\nb\rc")), "a b c");
        assert_eq!(sanitize_field(None), "");
    }

    #[test]
    fn collapse_whitespace_works() {
        assert_eq!(collapse_whitespace("  a   b  c  "), "a b c");
    }

    #[test]
    fn has_required_columns_works() {
        let mut row = HashMap::new();
        row.insert("a".to_string(), "1".to_string());
        row.insert("b".to_string(), "2".to_string());
        assert!(has_required_columns(&row, &["a", "b"]));
        assert!(!has_required_columns(&row, &["a", "c"]));
    }

    #[test]
    fn row_field_missing_column() {
        let mut row = HashMap::new();
        row.insert("a".to_string(), "1".to_string());
        assert_eq!(row_field(&row, "a"), "1");
        assert_eq!(row_field(&row, "missing"), "");
    }

    #[test]
    fn stream_json_array_basic() {
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("data.json");
        std::fs::write(&json_path, r#"{"meta": 1, "data": [[1, "ok"], [2, "skip"]]}"#).unwrap();
        let out_path = dir.path().join("out.txt");
        let mut render = |item: &Value| {
            if item[1] == "ok" {
                Some(format!("line={}", item[0]))
            } else {
                None
            }
        };
        let (written, skipped) =
            stream_json_array_to_txt(&json_path, &out_path, "data", &mut render).unwrap();
        assert_eq!(written, 1);
        assert_eq!(skipped, 1);
        assert_eq!(std::fs::read_to_string(&out_path).unwrap().trim(), "line=1");
    }

    #[test]
    fn stream_json_array_skips_unparseable_element() {
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("data.json");
        std::fs::write(&json_path, r#"{"data": [[1, "ok"], [2,], [3, "ok3"]]}"#).unwrap();
        let out_path = dir.path().join("out.txt");
        let mut render = |item: &Value| Some(format!("line={}", item[0]));
        let (written, skipped) =
            stream_json_array_to_txt(&json_path, &out_path, "data", &mut render).unwrap();
        assert_eq!(written, 2);
        assert_eq!(skipped, 1);
        let lines: Vec<String> = std::fs::read_to_string(&out_path)
            .unwrap()
            .lines()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(lines, vec!["line=1", "line=3"]);
    }

    #[test]
    fn stream_json_array_ignores_string_data_key() {
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("data.json");
        std::fs::write(
            &json_path,
            r#"{"data": "not an array", "meta": 1, "data": [[1, "ok"]]}"#,
        )
        .unwrap();
        let out_path = dir.path().join("out.txt");
        let mut render = |item: &Value| Some(format!("line={}", item[0]));
        let (written, skipped) =
            stream_json_array_to_txt(&json_path, &out_path, "data", &mut render).unwrap();
        assert_eq!(written, 1);
        assert_eq!(skipped, 0);
        assert_eq!(std::fs::read_to_string(&out_path).unwrap().trim(), "line=1");
    }

    #[test]
    fn stream_csv_short_row_keeps_missing_column_present() {
        // Python csv.DictReader fills a short row's missing column with restval=None
        // (present key); the row must NOT be skipped as missing-column.
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("in.csv");
        std::fs::write(&csv_path, "a,b,c\n1,ok\n2,ok,extra\n").unwrap();
        let out_path = dir.path().join("out.txt");
        let mut render = |row: &HashMap<String, String>| {
            // c is present (empty) on the short first row.
            Some(format!("a={} c={}", row["a"], row["c"]))
        };
        let (written, skipped) =
            stream_csv_to_txt(&csv_path, &out_path, &mut render, &["a", "b", "c"]).unwrap();
        assert_eq!(written, 2);
        assert_eq!(skipped, 0);
        let lines: Vec<String> = std::fs::read_to_string(&out_path)
            .unwrap()
            .lines()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(lines, vec!["a=1 c=", "a=2 c=extra"]);
    }

    #[test]
    fn stream_csv_basic() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("in.csv");
        std::fs::write(&csv_path, "a,b\n1,ok\n2,skip\n").unwrap();
        let out_path = dir.path().join("out.txt");
        let mut render = |row: &HashMap<String, String>| {
            if row.get("b").map(|s| s.as_str()) == Some("ok") {
                Some(format!("line={}", row["a"]))
            } else {
                None
            }
        };
        let (written, skipped) =
            stream_csv_to_txt(&csv_path, &out_path, &mut render, &["a", "b"]).unwrap();
        assert_eq!(written, 1);
        assert_eq!(skipped, 1);
        assert_eq!(std::fs::read_to_string(&out_path).unwrap().trim(), "line=1");
    }
}
