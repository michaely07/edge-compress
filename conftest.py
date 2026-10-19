import os

# Without a GPU, run Triton kernels in the CPU interpreter. This must be set before anything imports
# triton (torch/torchao can pull it in from other test modules), so it lives here, not in the test.
try:
    import torch
    _cuda = torch.cuda.is_available()
except Exception:
    _cuda = False
if not _cuda:
    os.environ.setdefault("TRITON_INTERPRET", "1")
