"""PyTorch reference backend (FP32, eager mode)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from edgebench.backends.base import validate_input


class TorchBackend:
    """Runs an in-memory ``nn.Module`` under ``torch.inference_mode``.

    ``torch.set_num_threads`` is process-wide; torch has no per-model thread setting.
    Inter-op threads are not set: ``torch.set_num_interop_threads`` fails once any
    parallel work has run in the process.
    """

    runtime = "torch"
    precision = "fp32"

    def __init__(self, module: nn.Module, *, num_threads: int, variant: str = "torch_fp32") -> None:
        torch.set_num_threads(num_threads)
        self.variant = variant
        self._module = module.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return float32 logits (N, C) for a float32 (N, 3, H, W) batch."""
        validate_input(x, self.variant)
        with torch.inference_mode():
            out = self._module(torch.from_numpy(x))
        return out.numpy().astype(np.float32, copy=False)
