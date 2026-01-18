"""
Implementation of the Lookahead optimizer wrapper.

Reference:
  Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back"
"""

from __future__ import annotations

from typing import Iterable

import torch


class Lookahead(torch.optim.Optimizer):
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        k: int = 5,
        alpha: float = 0.5,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid k: {k}")

        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.state = {}
        self.param_groups = self.base_optimizer.param_groups
        self._backup()
        self._step = 0

    def _backup(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    self.state[param] = {"slow_param": param.data.clone().detach()}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step += 1

        if self._step % self.k == 0:
            self.update_slow()

        return loss

    @torch.no_grad()
    def update_slow(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                slow = self.state[param]["slow_param"]
                slow.add_(param.data - slow, alpha=self.alpha)
                param.data.copy_(slow)

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    def state_dict(self):
        slow_params = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                     slow_params.append(self.state[param]["slow_param"])
        
        return {
            "base_state": self.base_optimizer.state_dict(),
            "slow_idx_map": slow_params, # Ordered list of slow params
            "step": self._step,
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base_state"])
        self._step = state_dict.get("step", 0)
        
        slow_params = state_dict.get("slow_idx_map", [])
        idx = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and idx < len(slow_params):
                    self.state[param]["slow_param"].copy_(slow_params[idx])
                    idx += 1
