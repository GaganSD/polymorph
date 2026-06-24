//! Real-model parity: the trained `mb_v0` ModernBERT pruner, run through the Rust
//! runtime (pure-Rust ModernBERT tokenizer + tract windowed inference + span
//! decode), must reproduce the Python reference (`bench/methods.py::LaMRMethod`).
//!
//! Gated on the 600 MB ONNX, which is gitignored (pull with
//! `modal volume get polymorph-lamr-v0 /out/mb_v0/onnx ...`). When absent the
//! test prints a skip line and passes, so CI without the artifact stays green.
//!
//! Fixture `tests/fixtures/mb_v0_forward_parity.json` is generated from the
//! checkpoint by the Python side (mb_ids + windowed drop-probs + the
//! `lamr+span@0.3` compressed string).

use std::path::PathBuf;

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("POLYMORPH_LAMR_MODEL") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/modal_out/mb_v0/onnx/model.onnx")
}

fn fixture() -> serde_json::Value {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mb_v0_forward_parity.json");
    serde_json::from_str(&std::fs::read_to_string(p).expect("read fixture")).expect("parse fixture")
}

#[test]
fn mb_v0_tokenizer_matches_python_on_fixture_text() {
    // Tokenizer parity on the exact fixture text (cheap; no model needed).
    let fx = fixture();
    let text = fx["text"].as_str().unwrap();
    let want: Vec<u32> = fx["mb_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let tok = polymorph::modernbert::get().expect("tokenizer");
    let (ids, _spans) = tok.encode_with_spans(text);
    assert_eq!(
        ids, want,
        "Rust ModernBERT ids must match Python encode_with_spans"
    );
}

#[test]
fn mb_v0_forward_probs_match_python() {
    let mp = model_path();
    if !mp.exists() {
        eprintln!(
            "SKIP mb_v0 forward parity: model not found at {}",
            mp.display()
        );
        return;
    }
    let fx = fixture();
    let text = fx["text"].as_str().unwrap();
    let want_probs: Vec<f32> = fx["probs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();

    let tok = polymorph::modernbert::get().expect("tokenizer");
    let (mb_ids, _spans) = tok.encode_with_spans(text);
    assert_eq!(
        mb_ids.len(),
        want_probs.len(),
        "token count must match fixture"
    );

    let model = polymorph::lamr::LamrOnnx::load(&mp).expect("load mb_v0 onnx");
    let got = model.forward_drop_probs(&mb_ids).expect("windowed forward");
    assert_eq!(got.len(), want_probs.len());

    // tract fp32 vs torch fp32, plus the fixed-512 export padding+masking the
    // partial last window (Python runs it un-padded). If the attention mask is
    // honoured these match tightly; allow a small numeric band.
    let mut max_abs = 0f32;
    for (a, b) in got.iter().zip(want_probs.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    eprintln!(
        "mb_v0 forward parity: max_abs_prob_diff = {max_abs:.6} over {} tokens",
        got.len()
    );
    assert!(
        max_abs < 2e-2,
        "max prob diff {max_abs} exceeds tolerance (windowing/padding mismatch?)"
    );
}

#[test]
fn mb_v0_compress_text_preserves_needle_and_shrinks() {
    let mp = model_path();
    if !mp.exists() {
        eprintln!("SKIP mb_v0 compress: model not found at {}", mp.display());
        return;
    }
    std::env::set_var("POLYMORPH_LAMR_MODEL", &mp);
    let fx = fixture();
    let text = fx["text"].as_str().unwrap();
    let grammars = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars");

    let res = polymorph::compress::compress_text(
        text,
        polymorph::Language::PlainText,
        &[],
        &grammars,
        Some(0.3),
        None,
    )
    .expect("compress_text");

    eprintln!(
        "compress_text: used_model={} in={} out={} ratio={:.2}",
        res.used_model, res.input_tokens, res.output_tokens, res.ratio
    );
    assert!(
        res.used_model,
        "the real ONNX model must have run (not the no-model passthrough)"
    );
    assert!(
        res.output_tokens < res.input_tokens,
        "compression must shrink the token count"
    );
    assert!(
        res.compressed.contains("DiskControllerFirmwareDeadlock"),
        "the error needle must survive compression"
    );
}

#[test]
fn e2e_ten_mb_log_compress_preserves_needles() {
    // The launch use case: a ~10 MB production log routed through the full
    // compress_log pipeline (dedup → lock → LaMR). Repetitive audit logs are the
    // ICP, so deterministic dedup collapses the bulk and the neural pruner runs on
    // a bounded residual — proving 10 MB is tractable and the needles survive.
    let mp = model_path();
    if !mp.exists() {
        eprintln!("SKIP 10MB e2e: model not found at {}", mp.display());
        return;
    }
    std::env::set_var("POLYMORPH_LAMR_MODEL", &mp);
    let grammars = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars");

    // Build ~10 MB: repetitive INFO/heartbeat chatter + a handful of needles.
    let mut log = String::with_capacity(11 * 1024 * 1024);
    let needles = [
        ("ERROR svc=payments txn=TX-88f3 status=503 err=DiskControllerFirmwareDeadlock host=10.4.2.9", "DiskControllerFirmwareDeadlock"),
        ("FATAL svc=auth user=alice@corp.io reason=TokenSignatureMismatch kid=KMS-7c1e", "TokenSignatureMismatch"),
        ("WARN svc=db pool=primary state_transition=LEADER->FOLLOWER term=4412", "LEADER->FOLLOWER"),
    ];
    let mut n = 0usize;
    while log.len() < 10 * 1024 * 1024 {
        log.push_str("2026-06-08T03:22:01Z [INFO] svc=api req=heartbeat path=/healthz status=200 dur=3ms region=us-east-1 pool=warm cache=hit\n");
        n += 1;
        if n.is_multiple_of(5000) {
            // sprinkle the needles through the stream
            let (line, _) = needles[(n / 5000) % needles.len()];
            log.push_str(line);
            log.push('\n');
        }
    }
    let orig_bytes = log.len();

    // Pipeline mirrors the compress_log tool: dedup, then neural compress.
    let plan = polymorph::dedup::dedup_plan(&log, polymorph::dedup::DedupOpts::default());
    eprintln!(
        "10MB e2e: {} bytes, dedup elided {} lines, reduced to {} bytes",
        orig_bytes,
        plan.elided_line_count(),
        plan.reduced.len()
    );
    let res = polymorph::compress::compress_text(
        &plan.reduced,
        polymorph::Language::PlainText,
        &[],
        &grammars,
        Some(0.3),
        Some(65_536),
    )
    .expect("compress_text");

    let orig_tokens = polymorph::tokenizer::count_tokens(&log).unwrap();
    let ratio = orig_tokens as f64 / res.output_tokens.max(1) as f64;
    eprintln!(
        "10MB e2e: used_model={} orig_tokens={} out_tokens={} overall_ratio={:.1}x",
        res.used_model, orig_tokens, res.output_tokens, ratio
    );
    assert!(res.used_model, "neural pruner must have run");
    assert!(
        ratio > 5.0,
        "a highly repetitive 10MB log should compress hugely (got {ratio:.1}x)"
    );
    for (_, needle) in &needles {
        assert!(
            res.compressed.contains(needle),
            "needle {needle:?} must survive the 10MB compression"
        );
    }
}

#[test]
fn plaintext_avoids_spurious_ast_locks_on_non_json() {
    // Regression for the lock-vs-prose budget bug: forcing `Json` on raw log text
    // makes tree-sitter ERROR-nodes lock ~half the doc; `PlainText` must lock
    // (almost) nothing so the pruner's budget isn't crowded onto free-text needles.
    let fx = fixture();
    let text = fx["text"].as_str().unwrap();
    let grammars = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars");

    let as_json =
        polymorph::compress_deterministic(text, polymorph::Language::Json, &[], &grammars).unwrap();
    let json_locked = as_json.mask.iter().filter(|&&m| m).count();

    let as_text =
        polymorph::compress_deterministic(text, polymorph::Language::PlainText, &[], &grammars)
            .unwrap();
    let text_locked = as_text.mask.iter().filter(|&&m| m).count();

    eprintln!(
        "locked: Json={json_locked}/{} PlainText={text_locked}/{}",
        as_json.mask.len(),
        as_text.mask.len()
    );
    assert!(
        json_locked > as_json.mask.len() / 4,
        "Json on non-JSON should over-lock (demonstrates the bug)"
    );
    assert_eq!(
        text_locked, 0,
        "PlainText with no keywords must lock nothing"
    );
}
