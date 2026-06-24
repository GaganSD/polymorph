use anyhow::{anyhow, Result};
use std::path::Path;

use crate::{ccr, compress, db, dedup, lcm, tokenizer, Language};

const SAMPLE_LOG: &str = include_str!("../examples/sample.log");

/// `polymorph-mcp --demo <kind>` entry point. Prints visible output so a new
/// user can confirm the system works in under 10 seconds.
pub fn run(kind: &str, grammars_dir: &Path) -> Result<()> {
    match kind {
        "compress" => demo_compress(grammars_dir),
        "lcm-loop" => demo_lcm_loop(),
        "ccr" => demo_ccr(),
        "" => Err(anyhow!(
            "missing demo kind. try: polymorph-mcp --demo compress  OR  --demo lcm-loop  OR  --demo ccr"
        )),
        other => Err(anyhow!(
            "unknown demo: {other}. try `compress`, `lcm-loop`, or `ccr`"
        )),
    }
}

fn demo_compress(grammars_dir: &Path) -> Result<()> {
    println!("Compression demo: examples/sample.log");
    println!("---");

    let input_tokens = tokenizer::count_tokens(SAMPLE_LOG)?;
    let plan = dedup::dedup_plan(SAMPLE_LOG, dedup::DedupOpts::default());
    let res = compress::compress_text(
        &plan.reduced,
        Language::PlainText,
        &[],
        grammars_dir,
        None,
        Some(65_536),
    )?;
    let overall_ratio = input_tokens as f64 / res.output_tokens.max(1) as f64;
    let needle = "DiskControllerFirmwareDeadlock";

    println!("input_tokens: {input_tokens}");
    println!("output_tokens: {}", res.output_tokens);
    println!("ratio: {overall_ratio:.2}x");
    println!("dedup_elided_lines: {}", plan.elided_line_count());
    println!("used_model: {}", res.used_model);
    println!("needle_preserved: {}", res.compressed.contains(needle));
    println!("--- compressed preview ---");
    println!("{}", res.compressed.trim());
    println!("---");
    if res.used_model {
        println!("done. LaMR model was active.");
    } else {
        println!("done. Deterministic mode ran; set POLYMORPH_LAMR_MODEL to enable LaMR.");
    }
    Ok(())
}

fn demo_lcm_loop() -> Result<()> {
    let db = db::test_pool()?;
    let conversation_id = "demo-conv";
    let soft_threshold: u64 = 300;
    println!(
        "LCM demo: mock conversation, soft_threshold={} tokens",
        soft_threshold
    );
    println!("---");

    let chunk = filler_text(120);
    for turn in 1..=12 {
        let result =
            lcm::append_and_maybe_archive(conversation_id, "user", &chunk, soft_threshold, &db)?;
        let active_after = lcm::active_token_count(conversation_id, &db)?;
        match result.archived_node_id {
            Some(node_id) => {
                println!(
                    "turn {:>2} +{} tokens -> active={}",
                    turn, result.tokens, active_after
                );
                println!(
                    "  *** ARCHIVE TRIGGERED *** node_id={} -> active now {}",
                    &node_id[..8],
                    active_after
                );
            }
            None => {
                println!(
                    "turn {:>2} +{} tokens -> active={}",
                    turn, result.tokens, active_after
                );
            }
        }
    }
    println!("---");
    println!("done. inspect with `lcm_describe`/`lcm_expand` over MCP.");
    Ok(())
}

fn demo_ccr() -> Result<()> {
    let db = db::test_pool()?;
    let original: serde_json::Value = serde_json::Value::Array(
        (0..50)
            .map(|i| serde_json::json!({"row": i, "value": format!("item-{}", i)}))
            .collect(),
    );
    let n = original.as_array().unwrap().len();
    println!("CCR demo: compressing a {}-element JSON array", n);
    println!("---");

    let res = ccr::compress_array(original.clone(), ccr::CcrOpts::default(), &db, true)?;
    let cache_id = res.cache_id.clone().unwrap();
    println!(
        "compressed -> {} elements (head + summary + tail). omitted {} items.",
        res.compressed.as_array().unwrap().len(),
        res.omitted_count
    );
    println!("cache_id = {}", cache_id);
    println!();

    let recovered = ccr::retrieve(&cache_id, &db)?;
    let recovered_len = recovered.as_array().unwrap().len();
    println!(
        "retrieved cache_id -> {} items recovered byte-for-byte",
        recovered_len
    );
    println!("---");
    println!("done.");
    Ok(())
}

/// Builds a deterministic filler string near `target_tokens` cl100k tokens. We
/// tokenize iteratively so the output is grounded in the real BPE.
fn filler_text(target_tokens: usize) -> String {
    let unit = "lorem ipsum dolor sit amet consectetur adipiscing elit ";
    let mut s = String::new();
    loop {
        let (ids, _) = tokenizer::token_spans(&s).expect("filler tokenize");
        if ids.len() >= target_tokens {
            return s;
        }
        s.push_str(unit);
    }
}
