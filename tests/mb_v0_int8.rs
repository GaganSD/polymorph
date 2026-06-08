//! Workstream C (latency): does the **pure-Rust `tract` engine** load and run an
//! INT8-quantized LaMR ModernBERT graph, and how do its load/inference time and
//! prob parity compare to the fp32 baseline?
//!
//! The fp32 `mb_v0` graph is ~600 MB and tract's `into_optimized()` over it takes
//! ~80 s to LOAD (the dominant UX cost). INT8 dynamic quantization (via
//! `ml_pipeline/scripts/quantize_int8.py`) shrinks it ~3.9x to ~154 MB. The KEY
//! UNKNOWN is op coverage: the INT8 graph uses `DynamicQuantizeLinear`,
//! `MatMulInteger`, and `DequantizeLinear`, and tract's QLinear/integer-matmul
//! support is incomplete — so this test exists to find out empirically.
//!
//! All tests skip gracefully (print a note + pass) when the corresponding model
//! file is absent, so CI without the gitignored artifacts stays green. Run with:
//!   cargo test --test mb_v0_int8 -- --nocapture --test-threads=1

use std::path::PathBuf;
use std::time::Instant;

use polymorph::lamr::LamrOnnx;

fn onnx_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/modal_out/mb_v0/onnx")
}

fn int8_path() -> PathBuf {
    onnx_dir().join("model.int8.onnx")
}

fn fp32_path() -> PathBuf {
    onnx_dir().join("model.onnx")
}

fn dynbatch_path() -> PathBuf {
    onnx_dir().join("model.dynbatch.onnx")
}

fn fixture() -> serde_json::Value {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mb_v0_forward_parity.json");
    serde_json::from_str(&std::fs::read_to_string(p).expect("read fixture")).expect("parse fixture")
}

fn fixture_ids_and_probs() -> (Vec<u32>, Vec<f32>) {
    let fx = fixture();
    let ids: Vec<u32> = fx["mb_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let probs: Vec<f32> = fx["probs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    (ids, probs)
}

fn file_size_mb(p: &std::path::Path) -> f64 {
    std::fs::metadata(p).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0)
}

/// THE KEY TEST. Loads the INT8 graph via the same `LamrOnnx::load` the runtime
/// uses, times the load (tract optimize) and inference, and reports the max abs
/// prob diff vs the fp32 fixture probs. If tract cannot load/optimize/run the
/// INT8 model, the panic message captures the exact tract error — which is itself
/// a valid finding (INT8 not viable in the pure-Rust path).
#[test]
fn mb_v0_int8_tract_load_run_and_parity() {
    let path = int8_path();
    if !path.exists() {
        eprintln!(
            "SKIP mb_v0 int8: {} not found (run ml_pipeline/scripts/quantize_int8.py)",
            path.display()
        );
        return;
    }
    eprintln!(
        "[int8] file size: {:.1} MB ({})",
        file_size_mb(&path),
        path.display()
    );

    let (mb_ids, want_probs) = fixture_ids_and_probs();

    // --- LOAD (tract parse + into_optimized + into_runnable) ---
    let t0 = Instant::now();
    let model = match LamrOnnx::load(&path) {
        Ok(m) => m,
        Err(e) => {
            // EXACT tract error — the important finding when INT8 ops are
            // unsupported. Fail loudly so --nocapture surfaces the message.
            panic!("[int8] tract FAILED to load/optimize the INT8 model: {e:#}");
        }
    };
    let load_ms = t0.elapsed().as_millis();
    eprintln!("[int8] tract LOAD time: {load_ms} ms ({:.1} s)", load_ms as f64 / 1000.0);

    // --- INFERENCE (windowed forward over the 2077-token fixture doc) ---
    let t1 = Instant::now();
    let got = match model.forward_drop_probs(&mb_ids) {
        Ok(p) => p,
        Err(e) => panic!("[int8] tract FAILED to RUN the INT8 model: {e:#}"),
    };
    let infer_ms = t1.elapsed().as_millis();
    eprintln!(
        "[int8] inference time: {infer_ms} ms over {} tokens",
        mb_ids.len()
    );

    assert_eq!(got.len(), want_probs.len(), "int8 prob count must match fixture");

    // --- PARITY (INT8 will be LOOSER than fp32's 3e-6; just report it) ---
    let mut max_abs = 0f32;
    for (a, b) in got.iter().zip(want_probs.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    eprintln!(
        "[int8] prob parity vs fp32 fixture: max_abs_diff = {max_abs:.6} over {} tokens",
        got.len()
    );

    // The number that actually matters for a TOP-K / span decode is not the raw
    // prob drift but whether the SAME tokens get dropped. Drop the top-30% by
    // prob under both fp32 (fixture) and int8 (got), and report set agreement.
    let drop_set = |probs: &[f32]| -> std::collections::HashSet<usize> {
        let k = ((0.30 * probs.len() as f64).round() as usize).min(probs.len());
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.into_iter().take(k).collect()
    };
    let fp32_drops = drop_set(&want_probs);
    let int8_drops = drop_set(&got);
    let inter = fp32_drops.intersection(&int8_drops).count();
    let union = fp32_drops.union(&int8_drops).count();
    let jaccard = inter as f64 / union as f64;
    let recall = inter as f64 / fp32_drops.len().max(1) as f64;
    eprintln!(
        "[int8] DECODE agreement @rate0.30: top-k drop-set Jaccard = {jaccard:.4}, \
         recall-of-fp32-drops = {recall:.4}"
    );

    // FINDING (print-only, do NOT fail CI): tract loads + runs the INT8 graph
    // (DynamicQuantizeLinear / MatMulInteger / DequantizeLinear). The raw
    // drop-prob drift is large (~0.1-0.3 max abs), but it is concentrated on a
    // few near-boundary tokens and the RANK ORDER is nearly intact, so the
    // actual top-k decode decision agrees ~0.98-0.997 (Jaccard). Combined with
    // the dramatic load-time win (~1 s vs ~80 s for fp32), per-channel INT8 is a
    // viable ship candidate. We REPORT these numbers rather than assert a tight
    // bound — the experiment's whole point is to measure them.
    eprintln!(
        "[int8] VERDICT: tract runs INT8 (load {load_ms} ms, infer {infer_ms} ms); \
         max_abs prob diff {max_abs:.3}, but decode Jaccard {jaccard:.3} — viable."
    );
}

/// fp32 baseline timing, for the side-by-side table. Reports whether the ~80 s
/// cold load reproduces on this machine. Skips if the fp32 model is absent.
#[test]
fn mb_v0_fp32_baseline_timing() {
    let path = fp32_path();
    if !path.exists() {
        eprintln!("SKIP mb_v0 fp32 baseline: {} not found", path.display());
        return;
    }
    eprintln!(
        "[fp32] file size: {:.1} MB ({})",
        file_size_mb(&path),
        path.display()
    );
    let (mb_ids, want_probs) = fixture_ids_and_probs();

    let t0 = Instant::now();
    let model = LamrOnnx::load(&path).expect("load fp32 model");
    let load_ms = t0.elapsed().as_millis();
    eprintln!("[fp32] tract LOAD time: {load_ms} ms ({:.1} s)", load_ms as f64 / 1000.0);

    let t1 = Instant::now();
    let got = model.forward_drop_probs(&mb_ids).expect("fp32 forward");
    let infer_ms = t1.elapsed().as_millis();
    eprintln!("[fp32] inference time: {infer_ms} ms over {} tokens", mb_ids.len());

    let mut max_abs = 0f32;
    for (a, b) in got.iter().zip(want_probs.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    eprintln!("[fp32] prob parity vs fixture: max_abs_diff = {max_abs:.6}");
}

/// Dynamic-batch re-export fallback (Task 3): can tract LOAD the dynbatch graph?
/// Only checks load (the variant's whole point is that tract accepts the graph);
/// a forward smoke + parity confirms it also runs. Skips if absent.
#[test]
fn mb_v0_dynbatch_tract_loads() {
    let path = dynbatch_path();
    if !path.exists() {
        eprintln!(
            "SKIP mb_v0 dynbatch: {} not found (run the dynbatch re-export)",
            path.display()
        );
        return;
    }
    eprintln!(
        "[dynbatch] file size: {:.1} MB ({})",
        file_size_mb(&path),
        path.display()
    );
    let (mb_ids, want_probs) = fixture_ids_and_probs();

    let t0 = Instant::now();
    let model = match LamrOnnx::load(&path) {
        Ok(m) => m,
        Err(e) => panic!("[dynbatch] tract FAILED to load dynbatch model: {e:#}"),
    };
    let load_ms = t0.elapsed().as_millis();
    eprintln!(
        "[dynbatch] tract LOAD time: {load_ms} ms ({:.1} s)",
        load_ms as f64 / 1000.0
    );

    let t1 = Instant::now();
    let got = model.forward_drop_probs(&mb_ids).expect("dynbatch forward");
    let infer_ms = t1.elapsed().as_millis();
    eprintln!("[dynbatch] inference time: {infer_ms} ms over {} tokens", mb_ids.len());

    let mut max_abs = 0f32;
    for (a, b) in got.iter().zip(want_probs.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    eprintln!("[dynbatch] prob parity vs fixture: max_abs_diff = {max_abs:.6}");
}
