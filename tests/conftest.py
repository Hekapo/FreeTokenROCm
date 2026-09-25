import sys

import pytest


@pytest.fixture(autouse=True, scope="session")
def _drain_gpu_at_session_end():
    """A process that exits with GPU work still queued can hang at shutdown on Windows/ROCm."""
    yield
    torch = sys.modules.get("torch")  # never import torch just for this
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


@pytest.fixture(autouse=True)
def _default_quant_backend():
    """Tests that install kernel requests must not leak them into the next test."""
    yield
    from freetoken.layers.quantization import QuantBackend, set_quant_backend

    set_quant_backend(QuantBackend())
