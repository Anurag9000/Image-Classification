"""
Adapters for parameter-efficient fine-tuning (LoRA / IA3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    train_base: bool = False
    target_modules: Sequence[str] = ("qkv", "kv", "proj", "fc", "mlp")


class LoRAInjectedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: int, train_base: bool) -> None:
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.train_base = train_base

        if not self.train_base:
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))

        self.scaling = alpha / rank
        self.reset_parameters()

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_weight = torch.matmul(self.lora_B, self.lora_A)
        lora_out = F.linear(x, lora_weight, bias=None) * self.scaling
        return base_out + lora_out


@dataclass
class IA3Config:
    target_modules: Sequence[str] = ("qkv", "kv", "proj", "fc", "mlp")


class IA3Linear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear
        self.gate = nn.Parameter(torch.ones(linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return out * self.gate.unsqueeze(0)


def _get_parent_and_attr(module: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(model: nn.Module, config: LoRAConfig) -> None:
    targets: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in config.target_modules):
            targets.append((name, module))

    for name, module in targets:
        parent, attr = _get_parent_and_attr(model, name)
        if isinstance(getattr(parent, attr), LoRAInjectedLinear):
            continue
        setattr(parent, attr, LoRAInjectedLinear(module, config.rank, config.alpha, config.train_base))


def inject_ia3(model: nn.Module, config: IA3Config) -> None:
    targets: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in config.target_modules):
            targets.append((name, module))

    for name, module in targets:
        parent, attr = _get_parent_and_attr(model, name)
        if isinstance(getattr(parent, attr), IA3Linear):
            continue
        setattr(parent, attr, IA3Linear(module))
