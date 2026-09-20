"""torch-cpu-backward-nan-tail-guard core: detect and guard against seven
CPU backward kernels returning a DIFFERENT gradient at NaN depending
purely on tensor length versus the CPU's SIMD vector-block width
(pytorch/pytorch#195075, open, labeled "module: correctness (silent)").

Independently reproduced from scratch on this host (torch 2.14.0,
Apple M4/arm64 NEON, CPU-only -- byte-exact match to the issue's own
reported output for a length-9 all-NaN tensor):

    logit_backward eps=0.1  grad[0]=0.0   grad[8]=nan
    hardtanh                grad[0]=0.0   grad[8]=1.0
    hardshrink              grad[0]=0.0   grad[8]=1.0
    softshrink              grad[0]=0.0   grad[8]=1.0
    elu                     grad[0]=nan   grad[8]=1.0
    celu                    grad[0]=nan   grad[8]=1.0
    hardswish               grad[0]=nan   grad[8]=1.0

Root cause (per the issue, which cites exact ATen source lines): each of
these ops' CPU kernel is implemented via ``cpu_kernel_vec``, which runs a
*scalar* lambda over the tensor's trailing remainder elements and a
*vectorized* lambda (writen as SIMD compare/blend intrinsics) over the
leading full-width SIMD blocks. The vector lambda is written as the
logically-negated form of the scalar predicate -- a rewrite that is only
valid for ORDERED (non-NaN) comparisons. At a NaN element, the scalar
and vector forms silently diverge, so whether a given NaN sits inside a
full vector block or in the scalar tail (purely a function of the
tensor's total length versus the CPU's SIMD width -- 8 lanes for
float32/NEON on this host, potentially 8 or 16 for AVX2/AVX-512 on other
CPUs) decides whether its gradient reads back as ``0.0``, the original
``grad_output``, or ``nan`` -- with no error, warning, or documented
length-dependence anywhere.

The issue explicitly declines to say which answer (NaN-propagates vs
NaN-absorbed) is "correct": each op already has several *disagreeing*
implementations (CPU scalar lambda, CPU vector lambda, CUDA, the
forward-mode-AD formula in derivatives.yaml, and for logit_backward, the
Python decomposition). The issue's own source reading states "CUDA
implements the scalar form in all of these" -- so this guard's policy is
to canonicalize on the SCALAR lambda's semantics (matching CUDA, and
matching what every one of these ops already does for any tensor short
enough to have no vector block at all), applied via straightforward,
non-negated elementwise comparisons through ``torch.where`` -- so the
SAME formula runs at every position regardless of tensor length, closing
the length-dependence entirely rather than picking an arbitrary new
answer.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Tuple


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported. Kept as a distinct type so
    callers can distinguish "torch isn't installed" from an actual
    diagnostic failure."""


def _import_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Guard functions: length-independent reimplementations matching the
# SCALAR lambda's semantics at every position (see module docstring).
# ---------------------------------------------------------------------------


def safe_logit_backward(grad_output, self, eps=None):
    """Guard for the backward of ``torch.special.logit(self, eps)``.
    When ``eps`` is given, out-of-range inputs get zero gradient;
    otherwise ``grad_output / (self * (1 - self))`` -- computed via
    straightforward (non-negated) comparisons so a NaN ``self`` always
    resolves the same way regardless of tensor length or position."""
    torch_module = _import_torch()
    if eps is None:
        # Issue: "eps=None is unaffected (that branch falls back to NaN
        # on both sides)" -- no length-dependence to guard here.
        return grad_output / (self * (1 - self))
    lo = eps
    hi = 1.0 - eps
    out_of_range = (self < lo) | (self > hi)
    return torch_module.where(
        out_of_range,
        torch_module.zeros_like(grad_output),
        grad_output / (self * (1 - self)),
    )


def safe_hardtanh_backward(grad_output, self, min_val=-1.0, max_val=1.0):
    """Guard for ``F.hardtanh``'s backward: zero gradient outside
    ``[min_val, max_val]``, else pass ``grad_output`` through unchanged."""
    torch_module = _import_torch()
    out_of_range = (self <= min_val) | (self >= max_val)
    return torch_module.where(out_of_range, torch_module.zeros_like(grad_output), grad_output)


def safe_shrink_backward(grad_output, self, lambd=0.5):
    """Guard for ``F.hardshrink``/``F.softshrink``'s shared backward
    shape: zero gradient inside ``[-lambd, lambd]``, else pass
    ``grad_output`` through unchanged."""
    torch_module = _import_torch()
    inside = (self >= -lambd) & (self <= lambd)
    return torch_module.where(inside, torch_module.zeros_like(grad_output), grad_output)


def safe_elu_backward(grad_output, self, alpha=1.0):
    """Guard for ``F.elu``'s backward (input-based, unit scale):
    ``grad_output * alpha * exp(self)`` for ``self <= 0``, else
    ``grad_output`` unchanged."""
    torch_module = _import_torch()
    neg_branch = grad_output * alpha * torch_module.exp(self)
    return torch_module.where(self <= 0, neg_branch, grad_output)


def safe_celu_backward(grad_output, self, alpha=1.0):
    """Guard for ``F.celu``'s backward: ``grad_output * exp(self / alpha)``
    for ``self <= 0``, else ``grad_output`` unchanged."""
    torch_module = _import_torch()
    neg_branch = grad_output * torch_module.exp(self / alpha)
    return torch_module.where(self <= 0, neg_branch, grad_output)


def safe_hardswish_backward(grad_output, self):
    """Guard for ``F.hardswish``'s backward: zero below -3, the
    ``grad_output * (2*self + 3) / 6`` ramp strictly between -3 and 3,
    and ``grad_output`` unchanged at/above 3 (also the NaN fallback,
    matching the scalar lambda's trailing branch)."""
    torch_module = _import_torch()
    below = self <= -3.0
    ramp = (self > -3.0) & (self < 3.0)
    ramp_val = grad_output * (2.0 * self + 3.0) / 6.0
    return torch_module.where(below, torch_module.zeros_like(grad_output), torch_module.where(ramp, ramp_val, grad_output))


# ---------------------------------------------------------------------------
# Diagnosis: reproduce the REAL (unguarded, autograd-driven) length
# dependence against the currently installed torch build, then verify
# each safe_* guard is length-independent (position 0 == position -1,
# NaN-aware) at the same tensor lengths.
# ---------------------------------------------------------------------------

_OPS = (
    "logit_backward",
    "hardtanh_backward",
    "hardshrink_backward",
    "softshrink_backward",
    "elu_backward",
    "celu_backward",
    "hardswish_backward",
)


def _nan_aware_eq(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _unguarded_grad_pair(torch_module, op: str, length: int) -> Tuple[float, float]:
    import torch.nn.functional as F

    x = torch_module.full((length,), float("nan"), requires_grad=True)
    if op == "logit_backward":
        torch_module.special.logit(x, eps=0.1).sum().backward()
    elif op == "hardtanh_backward":
        F.hardtanh(x).sum().backward()
    elif op == "hardshrink_backward":
        F.hardshrink(x).sum().backward()
    elif op == "softshrink_backward":
        F.softshrink(x).sum().backward()
    elif op == "elu_backward":
        F.elu(x).sum().backward()
    elif op == "celu_backward":
        F.celu(x).sum().backward()
    elif op == "hardswish_backward":
        F.hardswish(x).sum().backward()
    else:  # pragma: no cover - exhaustive over _OPS
        raise ValueError(op)
    return x.grad[0].item(), x.grad[-1].item()


def _guarded_grad_pair(torch_module, op: str, length: int) -> Tuple[float, float]:
    x = torch_module.full((length,), float("nan"))
    grad_output = torch_module.ones((length,))
    if op == "logit_backward":
        g = safe_logit_backward(grad_output, x, eps=0.1)
    elif op == "hardtanh_backward":
        g = safe_hardtanh_backward(grad_output, x)
    elif op == "hardshrink_backward":
        g = safe_shrink_backward(grad_output, x, lambd=0.5)
    elif op == "softshrink_backward":
        g = safe_shrink_backward(grad_output, x, lambd=0.5)
    elif op == "elu_backward":
        g = safe_elu_backward(grad_output, x)
    elif op == "celu_backward":
        g = safe_celu_backward(grad_output, x)
    elif op == "hardswish_backward":
        g = safe_hardswish_backward(grad_output, x)
    else:  # pragma: no cover - exhaustive over _OPS
        raise ValueError(op)
    return g[0].item(), g[-1].item()


@dataclasses.dataclass
class NanTailCase:
    op: str
    length: int
    buggy_grad_first: float
    buggy_grad_last: float
    buggy_position_dependent: bool
    guard_grad_first: float
    guard_grad_last: float
    guard_consistent: bool


def _run_case(torch_module, op: str, length: int) -> NanTailCase:
    buggy_first, buggy_last = _unguarded_grad_pair(torch_module, op, length)
    guard_first, guard_last = _guarded_grad_pair(torch_module, op, length)
    return NanTailCase(
        op=op,
        length=length,
        buggy_grad_first=buggy_first,
        buggy_grad_last=buggy_last,
        buggy_position_dependent=not _nan_aware_eq(buggy_first, buggy_last),
        guard_grad_first=guard_first,
        guard_grad_last=guard_last,
        guard_consistent=_nan_aware_eq(guard_first, guard_last),
    )


def diagnose(lengths=(9, 17, 33)) -> Dict[str, Any]:
    """Reproduce the length-dependent NaN-gradient divergence from
    scratch against the currently installed torch build, for every
    affected op at several tensor lengths (covering common SIMD widths:
    NEON=4/8, AVX2=8, AVX-512=16), then verify each ``safe_*`` guard is
    length-independent at the same lengths. Never trusts a cached/prior
    result -- every call re-runs the actual repro."""
    torch_module = _import_torch()

    cases: List[NanTailCase] = [
        _run_case(torch_module, op, length) for length in lengths for op in _OPS
    ]

    any_bug_present = any(c.buggy_position_dependent for c in cases)
    guard_fully_effective = all(c.guard_consistent for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/195075"],
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_bug_present": any_bug_present,
        "guard_fully_effective": guard_fully_effective,
    }
