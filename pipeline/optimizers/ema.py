"""
Model Exponential Moving Average (EMA) utility.

EMA is widely used in state-of-the-art training pipelines to stabilize training
and improve generalization, especially when coupled with sharpness-aware methods.
"""

from __future__ import annotations

import copy
from typing import Iterable, Optional

import torch


class ModelEMA:
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
        skip_buffers: bool = False,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")

        self.decay = decay
        self.device = device
        self.skip_buffers = skip_buffers

        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        if self.device is not None:
            self.ema_model.to(self.device)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(model.named_parameters())

        for name, param in model_params.items():
            if not param.requires_grad:
                continue
            ema_params[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

        if not self.skip_buffers:
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(model_buffer.detach().to(ema_buffer.device))

    def to(self, device: torch.device):
        self.ema_model.to(device)
        self.device = device

    def state_dict(self):
        return {
            "decay": self.decay,
            "ema_state": self.ema_model.state_dict(),
            "device": str(self.device) if self.device is not None else None,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.ema_model.load_state_dict(state_dict["ema_state"])
        device = state_dict.get("device")
        if device is not None:
            self.to(torch.device(device))

