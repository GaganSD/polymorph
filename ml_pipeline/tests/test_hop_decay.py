"""Hop-decay kernels."""

import math

import pytest

from polymorph_lamr.label.hop_decay import exp_decay, get_kernel, linear_decay


def test_exp_decay_monotone_and_unit_at_zero():
    k = exp_decay(alpha=0.5)
    assert k(0) == 1.0
    prev = 1.0
    for h in range(1, 8):
        cur = k(h)
        assert cur < prev
        assert 0.0 < cur <= 1.0
        prev = cur


def test_exp_decay_alpha_zero_constant_one():
    k = exp_decay(alpha=0.0)
    assert all(k(h) == 1.0 for h in range(5))


def test_exp_decay_rejects_negative_alpha():
    with pytest.raises(ValueError):
        exp_decay(alpha=-0.1)


def test_linear_decay_hits_zero_at_max_hops():
    k = linear_decay(max_hops=4)
    assert k(0) == 1.0
    assert k(4) == 0.0
    assert k(10) == 0.0
    assert math.isclose(k(2), 0.5)


def test_linear_decay_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        linear_decay(max_hops=0)


def test_get_kernel_dispatch():
    e = get_kernel({"kernel": "exp", "alpha": 1.0})
    l = get_kernel({"kernel": "linear", "max_hops": 3})
    assert math.isclose(e(0), 1.0)
    assert math.isclose(l(3), 0.0)
    with pytest.raises(ValueError):
        get_kernel({"kernel": "fancy"})
