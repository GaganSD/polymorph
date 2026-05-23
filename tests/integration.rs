use std::path::PathBuf;

use polymorph::{lock_payload, Language};
use pretty_assertions::assert_eq;

fn grammars_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars")
}

#[test]
fn json_keyword_and_structure() {
    let text = r#"{"api_key":"sk-prod-123","note":"this is just a description"}"#;
    let res = lock_payload(
        text,
        Language::Json,
        &["sk-prod-123".to_string()],
        &grammars_dir(),
    )
    .unwrap();

    assert_eq!(res.mask.len(), res.token_spans.len());
    assert_eq!(res.mask.len(), res.token_ids.len());
    assert_eq!(res.token_spans.last().unwrap().1, text.len());

    let prose = "this is just a description";
    let pos = text.find(prose).unwrap();
    let any_unlocked = res
        .token_spans
        .iter()
        .enumerate()
        .any(|(i, &(s, e))| s >= pos && e <= pos + prose.len() && !res.mask[i]);
    assert!(any_unlocked, "expected unlocked tokens inside prose value");

    let kw_pos = text.find("sk-prod-123").unwrap();
    let kw_range = kw_pos..(kw_pos + "sk-prod-123".len());
    let any_kw_token = res
        .token_spans
        .iter()
        .enumerate()
        .filter(|(_, &(s, e))| s < kw_range.end && e > kw_range.start)
        .count();
    assert!(any_kw_token > 0);
    for (i, &(s, e)) in res.token_spans.iter().enumerate() {
        if s < kw_range.end && e > kw_range.start {
            assert!(res.mask[i], "keyword token {i} must be locked");
        }
    }
}

#[test]
fn python_def_and_docstring() {
    let text = "\
@decorator
def hello(name):
    \"\"\"This docstring is filler prose that should be unlocked.\"\"\"
    # comment body also unlocked
    return name
";
    let res = lock_payload(text, Language::Python, &[], &grammars_dir()).unwrap();

    let def_pos = text.find("def ").unwrap();
    let def_locked = res
        .token_spans
        .iter()
        .enumerate()
        .any(|(i, &(s, e))| s <= def_pos && e > def_pos && res.mask[i]);
    assert!(def_locked, "`def` token must be locked");

    let prose = "docstring is filler prose";
    let pos = text.find(prose).unwrap();
    let any_unlocked = res
        .token_spans
        .iter()
        .enumerate()
        .any(|(i, &(s, e))| s >= pos && e <= pos + prose.len() && !res.mask[i]);
    assert!(any_unlocked, "docstring prose must have unlocked tokens");

    let comment_pos = text.find("comment body").unwrap();
    let any_unlocked = res
        .token_spans
        .iter()
        .enumerate()
        .any(|(i, &(s, e))| {
            s >= comment_pos && e <= comment_pos + "comment body".len() && !res.mask[i]
        });
    assert!(any_unlocked, "comment body must have unlocked tokens");
}

#[test]
fn json_arrays_and_atoms() {
    let text = r#"{"a":[1,2,3],"b":true,"c":null}"#;
    let res = lock_payload(text, Language::Json, &[], &grammars_dir()).unwrap();
    // Every atom (1, 2, 3, true, null) must be locked. They're named atom nodes.
    for atom in ["1", "2", "3", "true", "null"] {
        let pos = text.find(atom).unwrap();
        let range = pos..(pos + atom.len());
        let any_locked = res
            .token_spans
            .iter()
            .enumerate()
            .any(|(i, &(s, e))| s < range.end && e > range.start && res.mask[i]);
        assert!(any_locked, "atom `{atom}` must be locked");
    }
}

#[test]
fn ast_intervals_overlap_some_locked_token() {
    let text = r#"{"a":[1,2,3]}"#;
    let res = lock_payload(text, Language::Json, &[], &grammars_dir()).unwrap();
    for &(a, b) in &res.ast_intervals {
        let any = res
            .token_spans
            .iter()
            .enumerate()
            .any(|(i, &(s, e))| s < b && e > a && res.mask[i]);
        assert!(any, "AST interval [{a},{b}) has no locked token");
    }
}

#[test]
fn mask_length_invariant_across_inputs() {
    let cases = [
        (r#"{}"#, Language::Json),
        (r#"[]"#, Language::Json),
        ("x = 1\n", Language::Python),
        ("def f():\n    pass\n", Language::Python),
        ("# only a comment\n", Language::Python),
    ];
    for (text, lang) in cases {
        let res = lock_payload(text, lang, &[], &grammars_dir()).unwrap();
        assert_eq!(res.mask.len(), res.token_spans.len());
        assert_eq!(res.mask.len(), res.token_ids.len());
        if !res.token_spans.is_empty() {
            assert_eq!(res.token_spans.last().unwrap().1, text.len());
        }
    }
}

#[test]
fn empty_text_produces_empty_mask() {
    let res = lock_payload("", Language::Json, &[], &grammars_dir()).unwrap();
    assert!(res.mask.is_empty());
    assert!(res.token_ids.is_empty());
    assert!(res.token_spans.is_empty());
}

#[test]
fn unsupported_language_handled_at_lib_level() {
    assert!(Language::from_str("rust").is_none());
    assert!(Language::from_str("json").is_some());
    assert!(Language::from_str("python").is_some());
}

#[test]
fn keyword_locks_multiple_occurrences() {
    let text = r#"{"a":"secret","b":"secret"}"#;
    let res = lock_payload(text, Language::Json, &["secret".to_string()], &grammars_dir()).unwrap();
    let secret_positions: Vec<usize> = text
        .match_indices("secret")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(secret_positions.len(), 2);
    for pos in secret_positions {
        let range = pos..(pos + "secret".len());
        let count: usize = res
            .token_spans
            .iter()
            .enumerate()
            .filter(|(_, &(s, e))| s < range.end && e > range.start)
            .filter(|(i, _)| res.mask[*i])
            .count();
        assert!(count > 0, "secret at {pos} not locked");
    }
}

#[test]
fn python_brackets_and_colons_locked() {
    let text = "x = [1, 2, 3]\n";
    let res = lock_payload(text, Language::Python, &[], &grammars_dir()).unwrap();
    for sym in ['[', ']', ',', '='] {
        let pos = text.find(sym).unwrap();
        let any_locked = res
            .token_spans
            .iter()
            .enumerate()
            .any(|(i, &(s, e))| s <= pos && e > pos && res.mask[i]);
        assert!(any_locked, "`{sym}` must be locked");
    }
}

#[test]
fn large_payload_smoke() {
    // ~10k tokens, exercises the sweep + DAAC at scale.
    let mut text = String::from("[");
    for i in 0..2000 {
        if i > 0 {
            text.push(',');
        }
        text.push_str(&format!("{}", i));
    }
    text.push(']');
    let res = lock_payload(&text, Language::Json, &[], &grammars_dir()).unwrap();
    assert_eq!(res.mask.len(), res.token_spans.len());
    assert_eq!(res.token_spans.last().unwrap().1, text.len());
    // Brackets and commas must be locked.
    let any_open = res
        .token_spans
        .iter()
        .enumerate()
        .any(|(i, &(s, e))| s == 0 && e == 1 && res.mask[i]);
    assert!(any_open, "opening `[` must be locked");
}
