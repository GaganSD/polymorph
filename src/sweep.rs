/// Two-pointer sweep-line intersector.
///
/// `token_spans` is naturally sorted (tokens are contiguous, non-overlapping,
/// monotonically increasing). `ast_intervals` must be sorted by start; callers
/// should merge overlapping intervals before passing them in.
///
/// We mark `mask[i] = true` iff token i's byte span overlaps any AST interval,
/// OR token i's index falls inside any DAAC token-index interval.
pub fn build_mask(
    n_tokens: usize,
    token_spans: &[(usize, usize)],
    ast_intervals: &[(usize, usize)],
    daac_token_intervals: &[(usize, usize)],
) -> Vec<bool> {
    debug_assert_eq!(n_tokens, token_spans.len());
    let mut mask = vec![false; n_tokens];

    // Pass 1: AST byte intervals × token byte spans.
    // Two-pointer: advance `j` (AST) while its end is <= token start (no overlap).
    // After advancing, the AST interval at j (if any) is the first that could overlap.
    let mut j: usize = 0;
    for (i, &(ts, te)) in token_spans.iter().enumerate() {
        while j < ast_intervals.len() && ast_intervals[j].1 <= ts {
            j += 1;
        }
        if j < ast_intervals.len() {
            let (as_, ae) = ast_intervals[j];
            // overlap iff as_ < te && ts < ae
            if as_ < te && ts < ae {
                mask[i] = true;
            }
        }
    }

    // Pass 2: DAAC token-index intervals.
    for &(s, e) in daac_token_intervals {
        let s = s.min(n_tokens);
        let e = e.min(n_tokens);
        for slot in &mut mask[s..e] {
            *slot = true;
        }
    }

    mask
}

/// Sort + merge overlapping byte intervals. Useful for AST output before the
/// sweep, since recursive walks emit nested intervals.
pub fn sort_and_merge(mut intervals: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if intervals.is_empty() {
        return intervals;
    }
    intervals.retain(|(a, b)| a < b);
    intervals.sort_unstable_by_key(|&(a, _)| a);
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(intervals.len());
    let mut cur = intervals[0];
    for &(a, b) in &intervals[1..] {
        if a <= cur.1 {
            cur.1 = cur.1.max(b);
        } else {
            out.push(cur);
            cur = (a, b);
        }
    }
    out.push(cur);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ast_overlap_marks_tokens() {
        // 4 tokens of 2 bytes each: spans (0,2)(2,4)(4,6)(6,8)
        let spans = vec![(0usize, 2), (2, 4), (4, 6), (6, 8)];
        let ast = vec![(3usize, 5)]; // touches tokens 1 and 2
        let daac: Vec<(usize, usize)> = vec![];
        let mask = build_mask(4, &spans, &ast, &daac);
        assert_eq!(mask, vec![false, true, true, false]);
    }

    #[test]
    fn daac_marks_index_range() {
        let spans = vec![(0usize, 2), (2, 4), (4, 6), (6, 8)];
        let ast: Vec<(usize, usize)> = vec![];
        let daac = vec![(1usize, 3)];
        let mask = build_mask(4, &spans, &ast, &daac);
        assert_eq!(mask, vec![false, true, true, false]);
    }

    #[test]
    fn combined() {
        let spans = vec![(0usize, 2), (2, 4), (4, 6), (6, 8)];
        let ast = vec![(0usize, 1)]; // marks token 0
        let daac = vec![(3usize, 4)]; // marks token 3
        let mask = build_mask(4, &spans, &ast, &daac);
        assert_eq!(mask, vec![true, false, false, true]);
    }

    #[test]
    fn merge_basic() {
        let r = sort_and_merge(vec![(5, 7), (1, 3), (2, 4), (10, 12)]);
        assert_eq!(r, vec![(1, 4), (5, 7), (10, 12)]);
    }

    #[test]
    fn merge_touching_merges() {
        let r = sort_and_merge(vec![(0, 5), (5, 10)]);
        assert_eq!(r, vec![(0, 10)]);
    }
}
