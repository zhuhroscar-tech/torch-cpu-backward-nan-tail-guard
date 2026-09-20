# torch-cpu-backward-nan-tail-guard

Detects and guards against a real, currently-open PyTorch CPU defect:
seven backward kernels return a **different gradient at a NaN input
element depending purely on tensor length** versus the CPU's SIMD
vector-block width — not on anything about the math itself.

Tracking issue: [pytorch/pytorch#195075](https://github.com/pytorch/pytorch/issues/195075)
(open, labeled `module: correctness (silent)`).

## The bug, in one example

```python
import torch
import torch.nn.functional as F

x = torch.full((9,), float("nan"), requires_grad=True)
F.hardtanh(x).sum().backward()
print(x.grad[0].item(), x.grad[8].item())
# 0.0 1.0   <-- same op, same NaN value, DIFFERENT gradient by position
```

Independently reproduced on this project's own CI (and on the
maintainer's Apple M4/arm64 NEON host, torch 2.14.0) — byte-exact match
to the upstream issue's own reported values for a length-9 all-NaN
tensor:

| op | grad at position 0 (vector block) | grad at position 8 (scalar tail) |
| --- | --- | --- |
| `logit_backward` (`eps=0.1`) | `0.0` | `nan` |
| `hardtanh_backward` | `0.0` | `1.0` |
| `hardshrink`/`softshrink` backward | `0.0` | `1.0` |
| `elu`/`celu` backward | `nan` | `1.0` |
| `hardswish_backward` | `nan` | `1.0` |

**Root cause** (per the upstream issue, which cites exact line numbers
in `aten/src/ATen/native/cpu/`): each kernel is implemented with
`cpu_kernel_vec`, which runs a *scalar* lambda over the tensor's
trailing remainder and a *vectorized* (SIMD compare/blend) lambda over
full-width blocks. The vector lambda is written as the logically
negated form of the scalar predicate — valid only for **ordered**
(non-NaN) comparisons. At a NaN element the two silently disagree, so
whether a NaN lands in a full SIMD block or the scalar tail (purely a
function of `len(tensor)` vs. the CPU's vector width — 8 lanes for
float32/NEON on Apple Silicon, potentially 8 or 16 for AVX2/AVX-512 on
x86) decides the answer. No error, warning, or documented length
dependence exists anywhere.

The upstream issue explicitly does not pick a "correct" answer — each
op already has several *disagreeing* implementations (CPU scalar vs.
CPU vector vs. CUDA vs. the forward-mode-AD formula) — but it does note
that CUDA implements the scalar form throughout. This guard therefore
canonicalizes on the **scalar lambda's semantics** (matching CUDA, and
matching what every one of these ops already does for any tensor short
enough to have no vector block at all), reimplemented with
non-negated, straightforward comparisons via `torch.where` so the exact
same formula runs regardless of tensor length or position.

## Install

```bash
pip install torch-cpu-backward-nan-tail-guard[torch]
```

(`torch` is an optional extra — the CLI degrades gracefully with a
clear error, exit code 2, if it isn't installed.)

## Usage

```bash
torch-cpu-backward-nan-tail-guard          # human-readable report
torch-cpu-backward-nan-tail-guard --json   # machine-readable report
torch-cpu-backward-nan-tail-guard --no-color
```

Exit code `0` means every guard is length-independent on this host's
installed torch build; `1` means at least one guard case diverged
(a bug in this project, not upstream — please file an issue); `2` means
torch isn't installed.

## Library usage

```python
from torch_cpu_backward_nan_tail_guard.core import (
    safe_hardtanh_backward,
    safe_shrink_backward,   # hardshrink and softshrink share this shape
    safe_elu_backward,
    safe_celu_backward,
    safe_hardswish_backward,
    safe_logit_backward,
)
```

Each `safe_*` function takes `(grad_output, self, ...)` matching the
native backward kernel's own signature and returns the length-
independent gradient — safe to use anywhere the native backward would
otherwise be called directly on a tensor that might contain NaN.

## What this guard does NOT claim

- It does not claim the scalar lambda's answer is mathematically
  "more correct" than the vector lambda's — the upstream issue is
  explicit that a PyTorch maintainer needs to make that call. This
  guard's job is **consistency**: the same well-defined answer
  regardless of tensor length, matching CUDA's existing behavior.
- It does not patch PyTorch's compiled kernels — these are pure-Python
  `torch.where`-based reimplementations at the call-site level, exactly
  like this fleet's other `torch-*-guard` projects (see
  `torch-linalg-nan-guard` for the same pattern).

## Testing

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest --cov=torch_cpu_backward_nan_tail_guard --cov-report=term-missing
```

Tests verify each guard (a) matches real `torch.autograd`-computed
gradients on ordinary finite input (proving it's a genuine drop-in, not
just internally consistent), and (b) is length-independent at a NaN
input across lengths spanning common SIMD widths (1, 7, 8, 9, 16, 17,
32, 33). One test directly reproduces the unguarded upstream bug via
real `torch.nn.functional` + autograd with no guard involved, so CI
itself is evidence the defect still exists upstream on the tested
runners.

## Limitations

- Covers exactly the seven ops named in pytorch/pytorch#195075. If a
  future PyTorch release changes these kernels' implementation, the
  "bug reproduced" diagnostic may report `NOT reproduced` — check the
  tracking issue for its current status before assuming this guard is
  now a no-op.
- Verified on CPU only (the upstream issue is CPU-specific; the
  vectorized/scalar kernel split doesn't apply to CUDA, whose kernels
  already use the scalar form per the issue).
- The exact tensor length at which the divergence appears depends on
  the host CPU's SIMD vector width (8 lanes for float32/NEON on Apple
  Silicon; 8 or 16 lanes for AVX2/AVX-512 on x86) — confirmed to differ
  between this project's own macOS (arm64/NEON) and Linux (x86) CI
  runners. `diagnose()` and the CLI scan multiple lengths (9, 17, 33)
  specifically so at least one reliably lands past the runner's actual
  vector-block boundary regardless of architecture.

---

# torch-cpu-backward-nan-tail-guard（中文说明）

检测并规避 PyTorch 中一个真实存在、目前仍未修复的 CPU 缺陷：七个反向传播
（backward）内核在输入为 NaN 时，返回的梯度值**仅因张量长度是否超过 CPU
的 SIMD 向量块宽度而不同**——与数学计算本身无关。

追踪的上游 issue：[pytorch/pytorch#195075](https://github.com/pytorch/pytorch/issues/195075)
（开放中，标记为 `module: correctness (silent)`）。

## 一个例子说明问题

```python
import torch
import torch.nn.functional as F

x = torch.full((9,), float("nan"), requires_grad=True)
F.hardtanh(x).sum().backward()
print(x.grad[0].item(), x.grad[8].item())
# 0.0 1.0   <-- 同一个算子、同一个 NaN 值，梯度却因位置不同而不同
```

已在本项目自身的 CI（以及作者的 Apple M4/arm64 NEON 主机，torch
2.14.0）上独立复现——对长度为 9 的全 NaN 张量，结果与上游 issue 报告的
数值逐字节一致。

**根本原因**（详见上游 issue 引用的具体源码行）：每个内核都通过
`cpu_kernel_vec` 实现，对张量末尾的余数部分使用*标量* lambda，对完整
的 SIMD 块使用*向量化*（SIMD 比较/混合）lambda。向量化 lambda 是标量
判断条件的逻辑取反形式——这种改写仅在**有序**（非 NaN）比较下才成立。
遇到 NaN 元素时两者会悄然分歧，因此 NaN 落在完整 SIMD 块内还是标量尾部
（完全取决于张量长度相对于 CPU 向量宽度的关系）决定了最终结果。全程没
有任何错误、警告或文档说明这种长度依赖性。

上游 issue 明确表示不会判定哪个答案"更正确"——每个算子已经存在多个互
相矛盾的实现（CPU 标量、CPU 向量化、CUDA、前向模式自动微分公式）——但
指出 CUDA 全程使用标量形式。因此，本项目的修复策略是统一采用**标量
lambda 的语义**（与 CUDA 一致，也与所有这些算子在张量长度不足以形成
向量块时的现有行为一致），通过 `torch.where` 以非取反、直接的比较方式
重新实现，确保无论张量长度或位置如何，都执行完全相同的公式。

## 安装

```bash
pip install torch-cpu-backward-nan-tail-guard[torch]
```

## 使用方法

```bash
torch-cpu-backward-nan-tail-guard          # 人类可读报告
torch-cpu-backward-nan-tail-guard --json   # 机器可读报告
torch-cpu-backward-nan-tail-guard --no-color
```

## 局限性

- 仅覆盖 pytorch/pytorch#195075 中提到的七个算子。
- 仅在 CPU 上验证（该问题本身是 CPU 特有的）。
