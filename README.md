# torch-cpu-backward-nan-tail-guard

This standalone repository has been consolidated into [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards).

Use the consolidated package instead:

```bash
python -m pip install -e ".[torch]"
torch-guard run cpu-backward-nan-tail
```

Python imports moved to the umbrella package:

```python
from torch_correctness_guards import safe_hardtanh_backward, safe_logit_backward
from torch_correctness_guards import diagnose_cpu_backward_nan_tail
```

The original functionality is preserved there as the `cpu-backward-nan-tail` guard. This repository is kept as an archival pointer only.
