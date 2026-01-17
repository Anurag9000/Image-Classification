"""
Sharpness-Aware Minimization (SAM) optimizer wrapper.

Implementation adapted from the official paper:
  Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization"
and verified against the reference implementation released by the authors.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch


class SAM(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer_cls: type[torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            params: Iterable of parameters to optimize.
            base_optimizer_cls: The optimizer class to wrap (e.g., torch.optim.SGD).
            rho: Neighborhood size for SAM perturbation.
            adaptive: Whether to use the adaptive variant (ASAM).
            *args, **kwargs: Arguments forwarded to the base optimizer.
        """
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups, *args, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("SAM does not support sparse gradients.")
                if group["adaptive"]:
                    grad = grad * param.abs()
                norms.append(grad.norm(p=2))

        if not norms:
            return torch.tensor(0.0, device=device)

        norm = torch.norm(torch.stack(norms), p=2)
        return norm

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        scale = self.param_groups[0]["rho"] / (grad_norm + 1e-12)

        for group in self.param_groups:
            adaptive = group["adaptive"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                eps = param.grad
                if adaptive:
                    eps = eps * param.abs()
                param.add_(eps, alpha=scale)
                self.state[param]["sam_eps"] = eps * scale

        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def second_step(
        self,
        zero_grad: bool = False,
        grad_scaler: Optional["torch.amp.GradScaler"] = None,
    ) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                eps = state.pop("sam_eps", None)
                if eps is not None:
                    param.sub_(eps)

        if grad_scaler is not None:
            grad_scaler.step(self.base_optimizer)
        else:
            self.base_optimizer.step()

        if zero_grad:
            self.base_optimizer.zero_grad()

    def step(self, closure: Optional[callable] = None) -> None:
        raise RuntimeError("SAM requires calling first_step and second_step explicitly.")

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()
