/// Monotonic two-pointer sweep-line intersector for token lock masks.
///
/// `token_spans` is sorted and non-overlapping. `ast_intervals` and
/// `daac_token_intervals` must be sorted by start and merged (no overlaps).
///
/// Returns `mask[i] == true` when token `i` overlaps an AST byte interval or
/// falls inside a DAAC token-index interval.
pub fn build_mask(
    n_tokens: usize,
    token_spans: &[(usize, usize)],
    ast_intervals: &[(usize, usize)],
    daac_token_intervals: &[(usize, usize)],
) -> Vec<bool> {
    debug_assert_eq!(n_tokens, token_spans.len());
    let mut mask = vec![false; n_tokens];

    let mut ast_j: usize = 0;
    let mut daac_j: usize = 0;

    for i in 0..n_tokens {
        let (ts, te) = token_spans[i];

        while ast_j < ast_intervals.len() && ast_intervals[ast_j].1 <= ts {
            ast_j += 1;
        }
        if ast_j < ast_intervals.len() {
            let (as_, ae) = ast_intervals[ast_j];
            if as_ < te && ts < ae {
                mask[i] = true;
                continue;
            }
        }

        while daac_j < daac_token_intervals.len() && daac_token_intervals[daac_j].1 <= i {
            daac_j += 1;
        }
        if daac_j < daac_token_intervals.len() {
            let (ds, de) = daac_token_intervals[daac_j];
            if ds <= i && i < de {
                mask[i] = true;
            }
        }
    }

    mask
}

/// Sort and merge overlapping byte or token-index intervals.
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
        let spans = vec![(0usize, 2), (2, 4), (4, 6), (6, 8)];
        let ast = vec![(3usize, 5)];
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
        let ast = vec![(0usize, 1)];
        let daac = vec![(3usize, 4)];
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

    #[test]
    fn overlapping_daac_intervals_merged() {
        let spans: Vec<(usize, usize)> = (0..8).map(|i| (i * 2, i * 2 + 2)).collect();
        let raw = vec![(1usize, 3), (2, 5)];
        let daac = sort_and_merge(raw);
        let mask = build_mask(8, &spans, &[], &daac);
        assert!(mask[1] && mask[2] && mask[3] && mask[4]);
        assert!(!mask[0]);
    }
}
