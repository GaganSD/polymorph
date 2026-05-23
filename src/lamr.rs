use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::RngCore;

/// Deterministic seed for the mock pruner. Real LaMR will be a neural model in
/// M3; this stand-in just gives us a stable ~30% drop signal to wire the
/// pipeline against.
pub const LAMR_SEED: u64 = 0xA5C3_B6D2_E91F_4471;

/// Target drop rate when the real model swaps in. The mock returns true (drop)
/// with this probability for each unlocked token.
pub const DEFAULT_DROP_RATE: f64 = 0.30;

/// Runs a "forward pass" over the unlocked token ids and returns a Vec<bool>
/// of the same length where `true` = drop. Deterministic given the const seed
/// so tests are reproducible across runs.
pub fn dummy_lamr_forward_pass(unlocked_tokens: &[u32]) -> Vec<bool> {
    dummy_lamr_forward_pass_seeded(unlocked_tokens, LAMR_SEED, DEFAULT_DROP_RATE)
}

/// Same as above but accepts a custom seed + rate. Used in tests for varying
/// the drop rate without mutating the const.
pub fn dummy_lamr_forward_pass_seeded(
    unlocked_tokens: &[u32],
    seed: u64,
    drop_rate: f64,
) -> Vec<bool> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Multiplicative folding of each token id into the RNG keeps the output
    // stable for the same input slice — the next_u64 stream alone would NOT
    // depend on the token ids, but we want the mock to behave like a model
    // that conditions on its input.
    let threshold = (drop_rate * (u32::MAX as f64)) as u32;
    unlocked_tokens
        .iter()
        .map(|tok| {
            let r = rng.next_u32() ^ tok.wrapping_mul(0x9E37_79B1);
            r < threshold
        })
        .collect()
}

/// Scatter model decisions for the unlocked token slice back onto the original
/// token stream. This is the stable handoff point for the future ONNX-backed
/// LaMR path: Python exports semantic/dependency emissions plus `head_weights`;
/// Rust will decode one weighted CRF over the unlocked slice and pass the
/// resulting tag=drop bits here.
///
/// Locked tokens (`lock_mask[i] == true`) are NEVER dropped — invariant:
/// `lock_mask[i] => !drop_mask[i]`.
pub fn scatter_unlocked_drop_bits(lock_mask: &[bool], drop_bits: &[bool]) -> Vec<bool> {
    let unlocked = lock_mask.iter().filter(|&&locked| !locked).count();
    assert_eq!(
        unlocked,
        drop_bits.len(),
        "drop_bits must match the number of unlocked tokens"
    );

    let mut drop_mask = vec![false; lock_mask.len()];
    let mut cursor = 0;
    for (i, &locked) in lock_mask.iter().enumerate() {
        if !locked {
            drop_mask[i] = drop_bits[cursor];
            cursor += 1;
        }
    }
    drop_mask
}

/// Apply the mock LaMR pruning to a M1 lock mask. For each i where
/// `lock_mask[i] == false` (= unlocked), the i-th unlocked token's drop bit
/// determines `drop_mask[i]`.
///
/// Returns the parallel `drop_mask` of length `lock_mask.len()`.
pub fn apply_lamr(token_ids: &[u32], lock_mask: &[bool]) -> Vec<bool> {
    debug_assert_eq!(token_ids.len(), lock_mask.len());
    // Collect the unlocked-token slice in index order.
    let unlocked: Vec<u32> = token_ids
        .iter()
        .zip(lock_mask.iter())
        .filter(|(_, &locked)| !locked)
        .map(|(&id, _)| id)
        .collect();
    let drop_bits = dummy_lamr_forward_pass(&unlocked);

    scatter_unlocked_drop_bits(lock_mask, &drop_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_calls() {
        let tokens: Vec<u32> = (0..1000).collect();
        let a = dummy_lamr_forward_pass(&tokens);
        let b = dummy_lamr_forward_pass(&tokens);
        assert_eq!(a, b);
    }

    #[test]
    fn drop_rate_within_expected_band() {
        let tokens: Vec<u32> = (0..10_000).collect();
        let mask = dummy_lamr_forward_pass(&tokens);
        let dropped = mask.iter().filter(|&&b| b).count();
        let rate = dropped as f64 / mask.len() as f64;
        assert!(
            (rate - 0.30).abs() < 0.05,
            "drop rate {rate} outside 0.30 ± 0.05"
        );
    }

    #[test]
    fn apply_lamr_never_drops_locked() {
        let token_ids: Vec<u32> = (0..500).collect();
        // Lock every 3rd token.
        let lock_mask: Vec<bool> = (0..500).map(|i| i % 3 == 0).collect();
        let drop_mask = apply_lamr(&token_ids, &lock_mask);
        assert_eq!(drop_mask.len(), token_ids.len());
        for i in 0..500 {
            if lock_mask[i] {
                assert!(!drop_mask[i], "locked token {i} was dropped");
            }
        }
    }

    #[test]
    fn scatter_unlocked_drop_bits_never_drops_locked() {
        let lock_mask = vec![true, false, true, false, false, true];
        let drop_bits = vec![true, false, true];
        let drop_mask = scatter_unlocked_drop_bits(&lock_mask, &drop_bits);
        assert_eq!(drop_mask, vec![false, true, false, false, true, false]);
        for (locked, dropped) in lock_mask.iter().zip(drop_mask.iter()) {
            if *locked {
                assert!(!*dropped);
            }
        }
    }

    #[test]
    fn apply_lamr_drop_rate_only_on_unlocked() {
        let token_ids: Vec<u32> = (0..10_000).collect();
        // Lock half.
        let lock_mask: Vec<bool> = (0..10_000).map(|i| i % 2 == 0).collect();
        let drop_mask = apply_lamr(&token_ids, &lock_mask);
        let dropped: usize = drop_mask.iter().filter(|&&b| b).count();
        let unlocked: usize = lock_mask.iter().filter(|&&b| !b).count();
        let rate = dropped as f64 / unlocked as f64;
        assert!(
            (rate - 0.30).abs() < 0.05,
            "drop rate {rate} outside 0.30 ± 0.05 on unlocked slice"
        );
    }

    #[test]
    fn apply_lamr_empty_input() {
        let drop_mask = apply_lamr(&[], &[]);
        assert!(drop_mask.is_empty());
    }

    #[test]
    fn apply_lamr_all_locked_drops_nothing() {
        let token_ids: Vec<u32> = (0..100).collect();
        let lock_mask: Vec<bool> = vec![true; 100];
        let drop_mask = apply_lamr(&token_ids, &lock_mask);
        assert!(drop_mask.iter().all(|&b| !b));
    }
}
