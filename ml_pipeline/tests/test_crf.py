"""CRF correctness on small synthetic tensors."""

import torch

from polymorph_lamr.model.crf import LinearChainCRF, NUM_TAGS


def test_nll_is_nonnegative():
    torch.manual_seed(0)
    b, t = 3, 7
    crf = LinearChainCRF()
    emissions = torch.randn(b, t, NUM_TAGS)
    tags = torch.randint(0, NUM_TAGS, (b, t))
    mask = torch.ones(b, t, dtype=torch.bool)
    # NLL = logZ - score(gold). Should be >= 0 in well-defined CRFs because
    # logZ ≥ score(gold) always.
    nll = crf.nll(emissions, tags, mask)
    assert nll.item() >= -1e-4


def test_viterbi_matches_argmax_when_transitions_uniform():
    crf = LinearChainCRF()  # transitions all-zero by construction
    torch.manual_seed(0)
    b, t = 2, 5
    emissions = torch.randn(b, t, NUM_TAGS)
    mask = torch.ones(b, t, dtype=torch.bool)
    decoded = crf.decode(emissions, mask)
    greedy = emissions.argmax(dim=-1).tolist()
    assert decoded == greedy


def test_weighted_nll_zero_weight_doesnt_break_gradients():
    torch.manual_seed(0)
    crf = LinearChainCRF()
    b, t = 1, 4
    emissions = torch.randn(b, t, NUM_TAGS, requires_grad=True)
    tags = torch.zeros(b, t, dtype=torch.long)
    mask = torch.ones(b, t, dtype=torch.bool)
    weights = torch.zeros(b, t)  # entirely zero — degenerate but allowed
    loss = crf.nll(emissions, tags, mask, weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert emissions.grad is not None


def test_reduction_none_returns_per_sequence_nlls():
    torch.manual_seed(0)
    crf = LinearChainCRF()
    emissions = torch.randn(2, 3, NUM_TAGS)
    tags = torch.zeros(2, 3, dtype=torch.long)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    nll = crf.nll(emissions, tags, mask, reduction="none")
    assert nll.shape == (2,)
    assert torch.all(nll >= -1e-4)


def test_all_pad_row_has_zero_nll():
    crf = LinearChainCRF()
    emissions = torch.randn(2, 3, NUM_TAGS)
    tags = torch.zeros(2, 3, dtype=torch.long)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    nll = crf.nll(emissions, tags, mask, reduction="none")
    assert torch.isfinite(nll).all()
    assert nll[1].item() == 0.0


def test_single_token_sequence_nll_is_finite():
    crf = LinearChainCRF()
    emissions = torch.randn(2, 1, NUM_TAGS, requires_grad=True)
    tags = torch.tensor([[0], [1]])
    mask = torch.ones(2, 1, dtype=torch.bool)
    loss = crf.nll(emissions, tags, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert emissions.grad is not None


def test_weights_affect_partition_not_only_gold_path():
    crf = LinearChainCRF()
    emissions = torch.tensor([[[2.0, -2.0], [1.0, -1.0]]])
    tags = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.ones(1, 2, dtype=torch.bool)
    zero_weighted = crf.nll(emissions, tags, mask, torch.zeros(1, 2))
    one_weighted = crf.nll(emissions, tags, mask, torch.ones(1, 2))
    assert not torch.isclose(zero_weighted, one_weighted)


def test_decode_with_batched_params_matches_single_batch():
    torch.manual_seed(0)
    emissions = torch.randn(2, 4, NUM_TAGS)
    mask = torch.ones(2, 4, dtype=torch.bool)
    transitions = torch.randn(2, NUM_TAGS, NUM_TAGS)
    start = torch.randn(2, NUM_TAGS)
    end = torch.randn(2, NUM_TAGS)
    decoded = LinearChainCRF.decode_with_params(emissions, mask, transitions, start, end)
    for i in range(2):
        single = LinearChainCRF.decode_with_params(
            emissions[i : i + 1],
            mask[i : i + 1],
            transitions[i : i + 1],
            start[i : i + 1],
            end[i : i + 1],
        )
        assert decoded[i] == single[0]
