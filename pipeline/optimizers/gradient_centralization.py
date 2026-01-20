"""
Utilities for applying Gradient Centralization (GC).

Reference:
  Yong et al., "Gradient Centralization: A New Optimization Technique for Deep Neural Networks"
"""

from __future__ import annotations

import torch
import functools
import types
from typing import Iterable, Any


def apply_gradient_centralization(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    Wrap the optimizer to apply gradient centralization on convolutional/fully connected layers.
    """

    def centralize() -> None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.dim() > 1:
                    param.grad.data.add_(-param.grad.data.mean(dim=tuple(range(1, param.grad.dim())), keepdim=True))

    original_step = optimizer.step

    @functools.wraps(original_step)
    def gc_step(self, *args: Any, **kwargs: Any) -> Any:
        centralize()
        return original_step(*args, **kwargs)

    optimizer.step = types.MethodType(gc_step, optimizer)  # type: ignore[assignment]
    return optimizer

