use anyhow::{anyhow, Result};
use std::path::Path;

use crate::{db, lamr, lock_payload, Language};

/// Runs install-critical checks plus locking scenarios and prints a PASS/FAIL summary.
pub fn run(grammars_dir: &Path) -> Result<()> {
    let mut failures: Vec<String> = Vec::new();

    if let Err(e) = grammars_case(grammars_dir) {
        failures.push(format!("grammars: {e}"));
    }
    if let Err(e) = db_case() {
        failures.push(format!("db: {e}"));
    }
    if let Err(e) = model_case() {
        failures.push(format!("model: {e}"));
    }
    if let Err(e) = json_case(grammars_dir) {
        failures.push(format!("json: {e}"));
    }
    if let Err(e) = python_case(grammars_dir) {
        failures.push(format!("python: {e}"));
    }
    if let Err(e) = invariants_case(grammars_dir) {
        failures.push(format!("invariants: {e}"));
    }

    if failures.is_empty() {
        println!("PASS: install checks + 3 locking scenarios");
        Ok(())
    } else {
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
        Err(anyhow!("selftest failed ({} failure(s))", failures.len()))
    }
}

fn grammars_case(g: &Path) -> Result<()> {
    if !g.is_dir() {
        return Err(anyhow!("{} is not a directory", g.display()));
    }
    for file in ["tree-sitter-json.wasm", "tree-sitter-python.wasm"] {
        let path = g.join(file);
        if !path.is_file() {
            return Err(anyhow!("missing {}", path.display()));
        }
    }
    println!("grammars: ok ({})", g.display());
    Ok(())
}

fn db_case() -> Result<()> {
    let path = db::default_path()?;
    let _db = db::open_pool(&path)?;
    println!("db: ok ({})", path.display());
    Ok(())
}

fn model_case() -> Result<()> {
    match lamr::model_path_status() {
        lamr::ModelPathStatus::Unset => {
            println!("model: unset (deterministic mode; compress_log returns used_model=false)");
            Ok(())
        }
        lamr::ModelPathStatus::Found(path) => {
            println!("model: found ({})", path.display());
            Ok(())
        }
        lamr::ModelPathStatus::Empty => Err(anyhow!("POLYMORPH_LAMR_MODEL is set but empty")),
        lamr::ModelPathStatus::Missing { raw, resolved } => Err(anyhow!(
            "POLYMORPH_LAMR_MODEL={raw:?} resolved to {}, but no file exists there",
            resolved.display()
        )),
    }
}

fn json_case(g: &Path) -> Result<()> {
    let text = r#"{"api_key":"sk-prod-123","note":"this is just a description"}"#;
    let res = lock_payload(text, Language::Json, &["sk-prod-123".to_string()], g)?;

    // Find the byte position of the prose substring; assert at least one of
    // its tokens is unlocked (false).
    let prose = "this is just a description";
    let pos = text
        .find(prose)
        .ok_or_else(|| anyhow!("prose not found in fixture"))?;
    let prose_range = pos..(pos + prose.len());
    let any_unlocked = res
        .token_spans
        .iter()
        .enumerate()
        .any(|(i, &(s, e))| s >= prose_range.start && e <= prose_range.end && !res.mask[i]);
    if !any_unlocked {
        return Err(anyhow!("expected at least one unlocked token inside prose"));
    }

    // The opening brace `{` byte 0 must be locked.
    if !res.mask[0] {
        return Err(anyhow!("opening brace token must be locked"));
    }

    // The keyword span must be fully locked.
    let kw_pos = text
        .find("sk-prod-123")
        .ok_or_else(|| anyhow!("keyword fixture missing"))?;
    let kw_range = kw_pos..(kw_pos + "sk-prod-123".len());
    let kw_tokens: Vec<usize> = res
        .token_spans
        .iter()
        .enumerate()
        .filter_map(|(i, &(s, e))| {
            if s < kw_range.end && e > kw_range.start {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if kw_tokens.is_empty() {
        return Err(anyhow!("no tokens overlap keyword"));
    }
    for i in kw_tokens {
        if !res.mask[i] {
            return Err(anyhow!("keyword token {i} is unlocked"));
        }
    }
    Ok(())
}

fn python_case(g: &Path) -> Result<()> {
    let text = "\
@decorator
def hello(name):
    \"\"\"This docstring is filler prose that should be unlocked.\"\"\"
    # comment body also unlocked
    return name
";
    let res = lock_payload(text, Language::Python, &[], g)?;

    // `def` should be locked.
    let def_pos = text.find("def ").ok_or_else(|| anyhow!("def not found"))?;
    let def_locked = res
        .token_spans
        .iter()
        .enumerate()
        .any(|(i, &(s, e))| s <= def_pos && e > def_pos && res.mask[i]);
    if !def_locked {
        return Err(anyhow!("`def` keyword token not locked"));
    }

    // Docstring prose: pick a word inside the docstring and check at least one
    // unlocked token covers it.
    let prose = "docstring is filler prose";
    let pos = text
        .find(prose)
        .ok_or_else(|| anyhow!("prose not in fixture"))?;
    let prose_range = pos..(pos + prose.len());
    let any_unlocked = res
        .token_spans
        .iter()
        .enumerate()
        .any(|(i, &(s, e))| s >= prose_range.start && e <= prose_range.end && !res.mask[i]);
    if !any_unlocked {
        return Err(anyhow!("expected unlocked tokens inside docstring prose"));
    }

    // Comment body should have unlocked tokens.
    let comment = "comment body also unlocked";
    let pos = text
        .find(comment)
        .ok_or_else(|| anyhow!("comment not in fixture"))?;
    let c_range = pos..(pos + comment.len());
    let any_unlocked = res
        .token_spans
        .iter()
        .enumerate()
        .any(|(i, &(s, e))| s >= c_range.start && e <= c_range.end && !res.mask[i]);
    if !any_unlocked {
        return Err(anyhow!("expected unlocked tokens inside comment"));
    }

    Ok(())
}

fn invariants_case(g: &Path) -> Result<()> {
    let text = r#"{"a":[1,2,3],"b":"text"}"#;
    let res = lock_payload(text, Language::Json, &["needle-not-present".to_string()], g)?;

    if res.mask.len() != res.token_spans.len() {
        return Err(anyhow!("mask length mismatch"));
    }
    if res.mask.len() != res.token_ids.len() {
        return Err(anyhow!("token id / mask length mismatch"));
    }

    // Every AST byte interval must overlap at least one locked token.
    for &(a, b) in &res.ast_intervals {
        let any_locked = res
            .token_spans
            .iter()
            .enumerate()
            .any(|(i, &(s, e))| s < b && e > a && res.mask[i]);
        if !any_locked {
            return Err(anyhow!("AST interval [{a},{b}) has no locked token"));
        }
    }

    // Verify final span ends at text length.
    if let Some(last) = res.token_spans.last() {
        if last.1 != text.len() {
            return Err(anyhow!("token spans don't cover full text"));
        }
    }
    Ok(())
}
