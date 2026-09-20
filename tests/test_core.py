"""Correctness tests for the seven safe_* backward guards and the
diagnose() function, against real torch autograd -- not mocked.

Design note: for guard-vs-real comparisons we cannot compare against
"the buggy autograd result" directly (that IS the length-dependent bug
under test), so instead each safe_* function is checked for two
properties independently:

1. LENGTH-INDEPENDENCE: the guard gives the identical result at
   position 0 (would be in a vector block for a long-enough tensor) and
   position -1 (always in the scalar tail) for an all-NaN input, at
   several lengths spanning common SIMD widths.
2. AGREEMENT WITH THE SCALAR LAMBDA'S OWN DOCUMENTED SEMANTICS on
   well-defined (non-NaN) input: the guard must be a genuine drop-in,
   not merely "consistent but wrong" -- verified against real
   autograd-computed gradients on ordinary finite input, which is
   short enough to have no full vector block and therefore already
   uses the scalar lambda unaffected by this bug.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from torch_cpu_backward_nan_tail_guard.core import (  # noqa: E402
    TorchUnavailableError,
    _import_torch,
    diagnose,
    safe_celu_backward,
    safe_elu_backward,
    safe_hardswish_backward,
    safe_hardtanh_backward,
    safe_logit_backward,
    safe_shrink_backward,
)


def _real_grad(fn, x_values):
    x = torch.tensor(x_values, dtype=torch.float64, requires_grad=True)
    fn(x).sum().backward()
    return x.grad.tolist()


# ---------------------------------------------------------------------------
# Agreement with real autograd on ordinary (short, non-NaN) input -- proves
# the guards are genuine drop-ins, not merely internally consistent.
# ---------------------------------------------------------------------------


def test_safe_hardtanh_backward_matches_real_autograd_on_finite_input():
    xs = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    expected = _real_grad(F.hardtanh, xs)
    x = torch.tensor(xs, dtype=torch.float64)
    grad_output = torch.ones(len(xs), dtype=torch.float64)
    got = safe_hardtanh_backward(grad_output, x).tolist()
    assert got == pytest.approx(expected)


def test_safe_shrink_backward_matches_hardshrink_real_autograd():
    xs = [-1.0, -0.5, -0.4, 0.0, 0.4, 0.5, 1.0]
    expected = _real_grad(F.hardshrink, xs)
    x = torch.tensor(xs, dtype=torch.float64)
    grad_output = torch.ones(len(xs), dtype=torch.float64)
    got = safe_shrink_backward(grad_output, x, lambd=0.5).tolist()
    assert got == pytest.approx(expected)


def test_safe_shrink_backward_matches_softshrink_real_autograd():
    xs = [-1.0, -0.5, -0.4, 0.0, 0.4, 0.5, 1.0]
    expected = _real_grad(F.softshrink, xs)
    x = torch.tensor(xs, dtype=torch.float64)
    grad_output = torch.ones(len(xs), dtype=torch.float64)
    got = safe_shrink_backward(grad_output, x, lambd=0.5).tolist()
    assert got == pytest.approx(expected)


def test_safe_elu_backward_matches_real_autograd():
    xs = [-2.0, -0.5, 0.0, 0.5, 2.0]
    expected = _real_grad(F.elu, xs)
    x = torch.tensor(xs, dtype=torch.float64)
    grad_output = torch.ones(len(xs), dtype=torch.float64)
    got = safe_elu_backward(grad_output, x).tolist()
    assert got == pytest.approx(expected)


def test_safe_celu_backward_matches_real_autograd():
    xs = [-2.0, -0.5, 0.0, 0.5, 2.0]
    expected = _real_grad(F.celu, xs)
    x = torch.tensor(xs, dtype=torch.float64)
    grad_output = torch.ones(len(xs), dtype=torch.float64)
    got = safe_celu_backward(grad_output, x).tolist()
    assert got == pytest.approx(expected)


def test_safe_hardswish_backward_matches_real_autograd():
    xs = [-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0]
    expected = _real_grad(F.hardswish, xs)
    x = torch.tensor(xs, dtype=torch.float64)
    grad_output = torch.ones(len(xs), dtype=torch.float64)
    got = safe_hardswish_backward(grad_output, x).tolist()
    assert got == pytest.approx(expected, abs=1e-9)


def test_safe_logit_backward_matches_real_autograd_with_eps():
    xs = [0.01, 0.3, 0.5, 0.7, 0.99]

    def fn(x):
        return torch.special.logit(x, eps=0.1)

    expected = _real_grad(fn, xs)
    x = torch.tensor(xs, dtype=torch.float64)
    grad_output = torch.ones(len(xs), dtype=torch.float64)
    got = safe_logit_backward(grad_output, x, eps=0.1).tolist()
    assert got == pytest.approx(expected)


def test_safe_logit_backward_eps_none_delegates_unchanged():
    xs = [0.2, 0.5, 0.8]
    x = torch.tensor(xs, dtype=torch.float64)
    grad_output = torch.ones(len(xs), dtype=torch.float64)
    got = safe_logit_backward(grad_output, x, eps=None).tolist()
    expected = (grad_output / (x * (1 - x))).tolist()
    assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Length-independence at a NaN input: the actual regression this repo
# guards against. Each of these would FAIL before the fix (position 0
# and position -1 disagree once length >= one full SIMD block).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 17, 32, 33])
def test_safe_hardtanh_backward_is_length_independent_at_nan(length):
    x = torch.full((length,), float("nan"))
    grad_output = torch.ones(length)
    g = safe_hardtanh_backward(grad_output, x)
    first, last = g[0].item(), g[-1].item()
    assert math.isnan(first) == math.isnan(last)
    if not math.isnan(first):
        assert first == last


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 17, 32, 33])
def test_safe_shrink_backward_is_length_independent_at_nan(length):
    x = torch.full((length,), float("nan"))
    grad_output = torch.ones(length)
    g = safe_shrink_backward(grad_output, x, lambd=0.5)
    first, last = g[0].item(), g[-1].item()
    assert math.isnan(first) == math.isnan(last)
    if not math.isnan(first):
        assert first == last


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 17, 32, 33])
def test_safe_elu_backward_is_length_independent_at_nan(length):
    x = torch.full((length,), float("nan"))
    grad_output = torch.ones(length)
    g = safe_elu_backward(grad_output, x)
    first, last = g[0].item(), g[-1].item()
    assert math.isnan(first) == math.isnan(last)
    if not math.isnan(first):
        assert first == last


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 17, 32, 33])
def test_safe_celu_backward_is_length_independent_at_nan(length):
    x = torch.full((length,), float("nan"))
    grad_output = torch.ones(length)
    g = safe_celu_backward(grad_output, x)
    first, last = g[0].item(), g[-1].item()
    assert math.isnan(first) == math.isnan(last)
    if not math.isnan(first):
        assert first == last


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 17, 32, 33])
def test_safe_hardswish_backward_is_length_independent_at_nan(length):
    x = torch.full((length,), float("nan"))
    grad_output = torch.ones(length)
    g = safe_hardswish_backward(grad_output, x)
    first, last = g[0].item(), g[-1].item()
    assert math.isnan(first) == math.isnan(last)
    if not math.isnan(first):
        assert first == last


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 17, 32, 33])
def test_safe_logit_backward_is_length_independent_at_nan(length):
    x = torch.full((length,), float("nan"))
    grad_output = torch.ones(length)
    g = safe_logit_backward(grad_output, x, eps=0.1)
    first, last = g[0].item(), g[-1].item()
    assert math.isnan(first) == math.isnan(last)
    if not math.isnan(first):
        assert first == last


# ---------------------------------------------------------------------------
# Regression test that would have FAILED before this repo's guard existed:
# demonstrates the real bug via unguarded torch.nn.functional autograd
# (proves this is a genuine, currently-reproducible upstream defect, not a
# hypothetical), then proves the guard closes it via diagnose().
# ---------------------------------------------------------------------------


def test_unguarded_hardtanh_backward_is_length_dependent_at_nan_on_this_host():
    """This is the ACTUAL upstream bug (pytorch/pytorch#195075), reproduced
    with real torch.nn.functional + autograd -- no guard involved. If this
    ever starts failing (i.e. first == last, or both non-NaN and equal), it
    means upstream PyTorch has silently fixed the length dependence, which
    should be investigated and reflected in this test/README, not silenced."""
    x = torch.full((9,), float("nan"), requires_grad=True)
    F.hardtanh(x).sum().backward()
    first, last = x.grad[0].item(), x.grad[8].item()
    # The documented bug: grad[0]=0.0 (vector block), grad[8]=1.0 (scalar
    # tail) -- i.e. NOT the same value, and not both NaN.
    assert not (math.isnan(first) and math.isnan(last))
    assert first != last, (
        "Expected the documented length-dependent divergence "
        "(pytorch/pytorch#195075) on this host's torch build; if this "
        "assertion now fails, the upstream bug is fixed on this host -- "
        "verify against the currently installed torch version and update "
        "this test/README rather than deleting the assertion."
    )


def test_diagnose_reproduces_bug_and_confirms_guard_effective():
    report = diagnose(lengths=(9, 17))
    assert report["torch_version"]
    assert len(report["issue_urls"]) == 1
    assert "195075" in report["issue_urls"][0]
    # On any host affected by the upstream bug, at least one case must show
    # the length-dependent divergence.
    assert report["any_bug_present"] is True
    # The guard must close the divergence in EVERY tested case.
    assert report["guard_fully_effective"] is True
    assert len(report["cases"]) == 2 * 7  # 2 lengths x 7 ops


def test_diagnose_default_lengths_cover_multiple_simd_widths():
    report = diagnose()
    lengths = sorted({c["length"] for c in report["cases"]})
    assert lengths == [9, 17, 33]


def test_torch_unavailable_error_is_distinct_type(monkeypatch):
    import torch_cpu_backward_nan_tail_guard.core as core_module

    def _boom():
        raise TorchUnavailableError("torch is required for diagnosis and guarding; install the 'torch' extra.")

    monkeypatch.setattr(core_module, "_import_torch", _boom)
    with pytest.raises(TorchUnavailableError):
        core_module.diagnose()
