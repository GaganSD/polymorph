//! Byte-level alignment of an original text to its compressed variant, projected
//! onto cl100k tokens as a keep/drop mask — the exact training-label source.
//!
//! Ported from the Python `polymorph_lamr.label.align`. The alignment is a
//! faithful port of CPython's `difflib.SequenceMatcher.get_matching_blocks` over
//! raw bytes with `autojunk=False` (no junk heuristic): contiguous surviving byte
//! runs keep every token that overlaps them, so a needle whose bytes survive in
//! the compressed text survives in the label (the high-recall objective).

use anyhow::Result;
use std::collections::HashMap;

use crate::tokenizer::token_spans;

// Minimum matched byte-run length honored when projecting onto tokens (suppress
// coincidental short runs; every real needle is a contiguous run of >= 3 bytes).
const MIN_MATCH_RUN: usize = 3;

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

fn build_b2j(b: &[u8]) -> HashMap<u8, Vec<usize>> {
    let mut b2j: HashMap<u8, Vec<usize>> = HashMap::new();
    for (j, &val) in b.iter().enumerate() {
        b2j.entry(val).or_default().push(j);
    }
    b2j
}

/// `difflib.SequenceMatcher.find_longest_match` with no junk set.
fn find_longest_match(
    a: &[u8],
    b: &[u8],
    b2j: &HashMap<u8, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let mut besti = alo;
    let mut bestj = blo;
    let mut bestsize = 0usize;
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for i in alo..ahi {
        let mut newj2len: HashMap<usize, usize> = HashMap::new();
        if let Some(js) = b2j.get(&a[i]) {
            for &j in js {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let prev = if j > 0 {
                    *j2len.get(&(j - 1)).unwrap_or(&0)
                } else {
                    0
                };
                let k = prev + 1;
                newj2len.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2len = newj2len;
    }
    // Extend by non-junk elements on each end (no junk set, so this is all).
    while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
        besti -= 1;
        bestj -= 1;
        bestsize += 1;
    }
    while besti + bestsize < ahi
        && bestj + bestsize < bhi
        && a[besti + bestsize] == b[bestj + bestsize]
    {
        bestsize += 1;
    }
    (besti, bestj, bestsize)
}

/// `difflib.SequenceMatcher.get_matching_blocks` over bytes, including the
/// adjacent-block merge and the trailing `(la, lb, 0)` sentinel.
pub fn get_matching_blocks(a: &[u8], b: &[u8]) -> Vec<(usize, usize, usize)> {
    let la = a.len();
    let lb = b.len();
    let b2j = build_b2j(b);
    let mut queue: Vec<(usize, usize, usize, usize)> = vec![(0, la, 0, lb)];
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = find_longest_match(a, b, &b2j, alo, ahi, blo, bhi);
        if k > 0 {
            blocks.push((i, j, k));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    blocks.sort();
    let mut i1 = 0usize;
    let mut j1 = 0usize;
    let mut k1 = 0usize;
    let mut non_adjacent: Vec<(usize, usize, usize)> = Vec::new();
    for (i2, j2, k2) in blocks {
        if i1 + k1 == i2 && j1 + k1 == j2 {
            k1 += k2;
        } else {
            if k1 > 0 {
                non_adjacent.push((i1, j1, k1));
            }
            i1 = i2;
            j1 = j2;
            k1 = k2;
        }
    }
    if k1 > 0 {
        non_adjacent.push((i1, j1, k1));
    }
    non_adjacent.push((la, lb, 0));
    non_adjacent
}

/// Align `original` to `compressed` at the byte level, then project the matched
/// byte runs onto cl100k tokens. Returns `(token_ids, spans, keep_mask)` where
/// `keep_mask[i]` is true iff token i is preserved by the teacher's compression.
pub fn derive_mask(
    original: &str,
    compressed: &str,
) -> Result<(Vec<u32>, Vec<(usize, usize)>, Vec<bool>)> {
    let (ids, spans) = token_spans(original)?;
    let o = original.as_bytes();
    let c = compressed.as_bytes();
    let mut matched = vec![0u8; o.len()];
    for (a0, _b0, size) in get_matching_blocks(o, c) {
        if size < MIN_MATCH_RUN {
            continue;
        }
        for m in matched.iter_mut().take(a0 + size).skip(a0) {
            *m = 1;
        }
    }
    let mut keep = Vec::with_capacity(spans.len());
    for &(s, e) in &spans {
        if e <= s {
            keep.push(false);
            continue;
        }
        let mcount: usize = matched[s..e].iter().map(|&m| m as usize).sum();
        let has_content = (s..e).any(|i| matched[i] == 1 && !is_ws(o[i]));
        let full = mcount == (e - s);
        keep.push(full || (has_content && 2 * mcount >= (e - s)));
    }
    Ok((ids, spans, keep))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::decode_tokens;

    #[test]
    fn matching_blocks_canonical_difflib_example() {
        // From the difflib docs: SequenceMatcher(None, "abxcd", "abcd")
        // -> [Match(0,0,2), Match(3,2,2), Match(5,4,0)].
        let blocks = get_matching_blocks(b"abxcd", b"abcd");
        assert_eq!(blocks, vec![(0, 0, 2), (3, 2, 2), (5, 4, 0)]);
    }

    #[test]
    fn identical_text_keeps_everything() {
        let s = "ERROR connection refused by upstream host alpha";
        let (ids, _spans, keep) = derive_mask(s, s).unwrap();
        assert_eq!(keep.len(), ids.len());
        assert!(keep.iter().all(|k| *k));
    }

    #[test]
    fn surviving_needle_is_kept_dropped_one_is_not() {
        let original = "alpha beta gamma delta epsilon zeta eta theta iota";
        // compressed keeps "gamma delta epsilon" but drops the tail prose.
        let compressed = "alpha gamma delta epsilon";
        let (ids, _spans, keep) = derive_mask(original, compressed).unwrap();
        let kept_ids: Vec<u32> = ids
            .iter()
            .zip(keep.iter())
            .filter(|(_, k)| **k)
            .map(|(id, _)| *id)
            .collect();
        let kept_text = decode_tokens(&kept_ids).unwrap();
        assert!(kept_text.contains("gamma delta epsilon"));
        assert!(!kept_text.contains("theta iota"));
    }
}
