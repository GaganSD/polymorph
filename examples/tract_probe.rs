//! tract op-compatibility probe for a candidate encoder ONNX.
//!
//! `into_optimized()` is where tract rejects ops it doesn't implement, so this is
//! the make-or-break test for swapping LaMR's from-scratch embedding for a
//! pretrained encoder (ModernBERT / DeBERTa) while keeping the Rust runtime.
//!
//! Usage: cargo run --example tract_probe -- path/to/model.onnx [seq_len]
//!
//! Reports: parse OK?  optimize OK (or the unsupported op)?  runnable OK?  and a
//! dummy forward pass over (input_ids:i64[1,T], attention_mask:i64[1,T]).

use tract_onnx::prelude::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: tract_probe <model.onnx> [seq_len]");
            std::process::exit(2);
        }
    };
    let seq_len: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);

    println!("[tract_probe] model: {path}  seq_len: {seq_len}");

    let parsed = match tract_onnx::onnx().model_for_path(&path) {
        Ok(m) => {
            println!("[tract_probe] parse: OK ({} nodes)", m.nodes().len());
            m
        }
        Err(e) => {
            println!("[tract_probe] parse: FAILED -> {e}");
            std::process::exit(1);
        }
    };

    let optimized = match parsed.clone().into_optimized() {
        Ok(m) => {
            println!("[tract_probe] optimize: OK ({} nodes)", m.nodes().len());
            m
        }
        Err(e) => {
            println!("[tract_probe] optimize: FAILED (unsupported op?) -> {e}");
            std::process::exit(1);
        }
    };

    let runnable = match optimized.into_runnable() {
        Ok(m) => {
            println!("[tract_probe] runnable: OK");
            m
        }
        Err(e) => {
            println!("[tract_probe] runnable: FAILED -> {e}");
            std::process::exit(1);
        }
    };

    // Dummy forward: input_ids + attention_mask, both i64 [1, seq_len].
    let ids = tract_ndarray::Array2::<i64>::from_shape_fn((1, seq_len), |(_, j)| (j as i64) % 100 + 1);
    let mask = tract_ndarray::Array2::<i64>::from_elem((1, seq_len), 1i64);
    match runnable.run(tvec!(ids.into_tensor().into(), mask.into_tensor().into())) {
        Ok(out) => {
            print!("[tract_probe] forward: OK; outputs:");
            for (i, o) in out.iter().enumerate() {
                print!(" [{i}]={:?}", o.shape());
            }
            println!();
            println!("[tract_probe] VERDICT: tract-compatible ✅");
        }
        Err(e) => {
            println!("[tract_probe] forward: FAILED -> {e}");
            println!("[tract_probe] VERDICT: loads but won't run as-given (check input dtypes/names)");
            std::process::exit(1);
        }
    }
}
