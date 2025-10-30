"""
Utilities for applying Gradient Centralization (GC).

Reference:
  Yong et al., "Gradient Centralization: A New Optimization Technique for Deep Neural Networks"
"""

from __future__ import annotations

import torch


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

    def gc_step(*args, **kwargs):
        centralize()
        return original_step(*args, **kwargs)

    optimizer.step = gc_step  # type: ignore[assignment]
    return optimizer

