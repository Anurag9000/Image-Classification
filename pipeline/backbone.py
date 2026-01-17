from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from .adapters import IA3Config, LoRAConfig, inject_ia3, inject_lora
from utils import apply_token_merging


class TokenLearner(nn.Module):
    def __init__(self, embed_dim: int, num_output_tokens: int) -> None:
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.attention = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_output_tokens),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attention(x)  # (B, N, K)
        return torch.einsum("bnk,bnc->bkc", attn, x)


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha
        self._beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x

        b, c, h, w = x.size()
        x = x.view(b, c, -1)
        mu = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + 1e-6)

        mu_shuffle = mu[torch.randperm(b)]
        sigma_shuffle = sigma[torch.randperm(b)]

        lam = self._beta.sample((b, 1, 1)).to(x.device)

        mu_mix = lam * mu + (1 - lam) * mu_shuffle
        sigma_mix = lam * sigma + (1 - lam) * sigma_shuffle

        x_norm = (x - mu) / sigma
        mixed = x_norm * sigma_mix + mu_mix
        return mixed.view(b, c, h, w)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduction_channels = max(channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduction_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channels, channels),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        channel_att = torch.sigmoid(self.fc(avg_pool) + self.fc(max_pool)).view(b, c, 1, 1)
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class DynamicHeadAttention(nn.Module):
    def __init__(self, input_dim: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        hidden = max(input_dim // reduction_ratio, 64)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        attn_weights = self.fc(combined)
        return combined * attn_weights


@dataclass
class BackboneConfig:
    cnn_model: str = "convnextv2_base"
    vit_model: str = "swinv2_base_window12_192_22k"
    pretrained: bool = True
    use_cbam: bool = True
    fusion_dim: int = 1024
    token_merging_ratio: Optional[float] = None
    token_learner_tokens: Optional[int] = None
    freeze_cnn: bool = False
    freeze_vit: bool = False
    lora_rank: Optional[int] = None
    lora_alpha: int = 16
    lora_train_base: bool = False
    lora_target_modules: Tuple[str, ...] = field(
        default_factory=lambda: ("qkv", "kv", "proj", "fc", "mlp")
    )
    use_ia3: bool = False
    ia3_target_modules: Tuple[str, ...] = field(
        default_factory=lambda: ("qkv", "kv", "proj", "fc", "mlp")
    )
    cnn_drop_path_rate: float = 0.0
    vit_drop_path_rate: float = 0.2
    mixstyle: bool = False
    mixstyle_p: float = 0.5
    mixstyle_alpha: float = 0.1


def _reset_classifier(model: nn.Module) -> None:
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0)
    elif hasattr(model, "head"):
        model.head = nn.Identity()


class HybridBackbone(nn.Module):
    def __init__(self, config: Optional[BackboneConfig] = None) -> None:
        super().__init__()
        self.cfg = config or BackboneConfig()

        self.cnn_backbone = create_model(
            self.cfg.cnn_model,
            pretrained=self.cfg.pretrained,
            drop_path_rate=self.cfg.cnn_drop_path_rate,
        )
        _reset_classifier(self.cnn_backbone)

        self.vit_backbone = create_model(
            self.cfg.vit_model,
            pretrained=self.cfg.pretrained,
            drop_path_rate=self.cfg.vit_drop_path_rate,
        )
        _reset_classifier(self.vit_backbone)

        if self.cfg.token_merging_ratio:
            self.vit_backbone = apply_token_merging(self.vit_backbone, ratio=self.cfg.token_merging_ratio)

        if self.cfg.lora_rank:
            inject_lora(
                self.vit_backbone,
                LoRAConfig(
                    rank=self.cfg.lora_rank,
                    alpha=self.cfg.lora_alpha,
                    train_base=self.cfg.lora_train_base,
                    target_modules=self.cfg.lora_target_modules,
                ),
            )

        if self.cfg.use_ia3:
            inject_ia3(
                self.vit_backbone,
                IA3Config(target_modules=self.cfg.ia3_target_modules),
            )

        if self.cfg.freeze_cnn:
            for param in self.cnn_backbone.parameters():
                param.requires_grad = False
        if self.cfg.freeze_vit and not self.cfg.lora_rank:
            for param in self.vit_backbone.parameters():
                param.requires_grad = False

        self.cnn_dim = getattr(self.cnn_backbone, "num_features", 1024)
        self.vit_dim = getattr(self.vit_backbone, "num_features", 1024)

        self.use_cbam = self.cfg.use_cbam
        if self.use_cbam:
            self.cbam = CBAM(self.cnn_dim)
        self.mixstyle_module = None
        if self.cfg.mixstyle:
            self.mixstyle_module = MixStyle(p=self.cfg.mixstyle_p, alpha=self.cfg.mixstyle_alpha)

        self.token_learner = None
        if self.cfg.token_learner_tokens:
            self.token_learner = TokenLearner(self.vit_dim, self.cfg.token_learner_tokens)

        total_dim = self.cnn_dim + self.vit_dim
        self.dha = DynamicHeadAttention(total_dim)
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_dim, self.cfg.fusion_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.cfg.fusion_dim),
        )

    def _cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn_backbone.forward_features(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if self.use_cbam and isinstance(feat, torch.Tensor) and feat.dim() == 4:
            feat = self.cbam(feat)
        if self.mixstyle_module and self.training and isinstance(feat, torch.Tensor) and feat.dim() == 4:
            feat = self.mixstyle_module(feat)
        if hasattr(self.cnn_backbone, "forward_head"):
            feat = self.cnn_backbone.forward_head(feat, pre_logits=True)
        else:
            if feat.dim() == 4:
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            else:
                feat = feat.flatten(1) if feat.dim() > 2 else feat
        return feat

    def _vit_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.vit_backbone.forward_features(x)
        if isinstance(feat, dict):
            feat = feat.get("x_norm_clstoken") or feat.get("x_norm", None) or feat.get("x", None)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        if self.token_learner and isinstance(feat, torch.Tensor) and feat.dim() == 3 and feat.size(1) > 1:
            cls_token, patch_tokens = feat[:, :1], feat[:, 1:]
            reduced = self.token_learner(patch_tokens)
            feat = torch.cat([cls_token, reduced], dim=1)
        if hasattr(self.vit_backbone, "forward_head"):
            feat = self.vit_backbone.forward_head(feat, pre_logits=True)
        else:
            if feat.dim() == 3:
                feat = feat[:, 0]
            elif feat.dim() == 4:
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            else:
                feat = feat.flatten(1) if feat.dim() > 2 else feat
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self._cnn_features(x)
        vit_feat = self._vit_features(x)

        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        refined = self.dha(combined)
        fused = self.fusion_fc(refined)
        return fused


if __name__ == "__main__":
    cfg = BackboneConfig(
        vit_model="dinov2_base14",
        lora_rank=8,
        token_merging_ratio=0.5,
        use_ia3=True,
    )
    model = HybridBackbone(cfg).cuda()
    dummy = torch.randn(2, 3, 224, 224).cuda()
    out = model(dummy)
    print("Output Shape:", out.shape)


