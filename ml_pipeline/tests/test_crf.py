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
