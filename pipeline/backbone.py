from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from .adapters import IA3Config, LoRAConfig, inject_ia3, inject_lora
from utils import apply_token_merging

import logging
LOGGER = logging.getLogger(__name__)


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

        idx = torch.randperm(b).to(x.device)
        mu_shuffle = mu[idx]
        sigma_shuffle = sigma[idx]

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
    
    # CNN Adaptation
    cnn_lora_rank: Optional[int] = None
    cnn_ia3: bool = False
    
    # ViT Adaptation
    vit_lora_rank: Optional[int] = None
    vit_ia3: bool = False
    
    # Shared Adaptation Params
    lora_alpha: int = 16
    lora_train_base: bool = False
    lora_target_modules: Tuple[str, ...] = field(
        default_factory=lambda: ("qkv", "kv", "proj", "fc", "mlp")
    )
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


import logging
LOGGER = logging.getLogger(__name__)

class HybridBackbone(nn.Module):
    def __init__(self, config: Optional[BackboneConfig] = None) -> None:
        super().__init__()
        self.cfg = config or BackboneConfig()

        self.token_learner = None  # Initialize early to avoid AttributeError in _vit_features during dim detection
        self.cnn_backbone = create_model(
            self.cfg.cnn_model,
            pretrained=self.cfg.pretrained,
            drop_path_rate=self.cfg.cnn_drop_path_rate,
        )
        _reset_classifier(self.cnn_backbone)

        if self.cfg.vit_model:
            self.vit_backbone = create_model(
                self.cfg.vit_model,
                pretrained=self.cfg.pretrained,
                drop_path_rate=self.cfg.vit_drop_path_rate,
            )
            _reset_classifier(self.vit_backbone)
            
            # DEBUG: Verify weights are healthy
            for name, param in self.vit_backbone.named_parameters():
                 if torch.isnan(param).any():
                       LOGGER.critical(f"CRITICAL: Found NaN in pretrained ViT weights! Layer: {name}")
                       raise RuntimeError(f"NaN in ViT weights: {name}")

            if self.cfg.token_merging_ratio:
                self.vit_backbone = apply_token_merging(self.vit_backbone, ratio=self.cfg.token_merging_ratio)
        else:
            self.vit_backbone = nn.Identity()



        if self.cfg.cnn_lora_rank:
            lora_cfg = LoRAConfig(
                rank=self.cfg.cnn_lora_rank,
                alpha=self.cfg.lora_alpha,
                train_base=self.cfg.lora_train_base,
                target_modules=self.cfg.lora_target_modules,
            )
            inject_lora(self.cnn_backbone, lora_cfg)
            # Ensure LoRA parameters are trainable even if CNN is frozen
            for n, p in self.cnn_backbone.named_parameters():
                if "lora_" in n or "gate" in n:
                    p.requires_grad = True

        if self.cfg.cnn_ia3:
            ia3_cfg = IA3Config(target_modules=self.cfg.ia3_target_modules)
            inject_ia3(self.cnn_backbone, ia3_cfg)
            # Ensure IA3 parameters are trainable
            for n, p in self.cnn_backbone.named_parameters():
                if "gate" in n:
                    p.requires_grad = True

        if self.cfg.vit_lora_rank and self.cfg.vit_model:
            lora_cfg = LoRAConfig(
                rank=self.cfg.vit_lora_rank,
                alpha=self.cfg.lora_alpha,
                train_base=self.cfg.lora_train_base,
                target_modules=self.cfg.lora_target_modules,
            )
            inject_lora(self.vit_backbone, lora_cfg)
            # Ensure LoRA parameters are trainable even if ViT is frozen
            for n, p in self.vit_backbone.named_parameters():
                if "lora_" in n or "gate" in n:
                    p.requires_grad = True

        if self.cfg.vit_ia3 and self.cfg.vit_model:
            ia3_cfg = IA3Config(target_modules=self.cfg.ia3_target_modules)
            inject_ia3(self.vit_backbone, ia3_cfg)
            # Ensure IA3 parameters are trainable
            for n, p in self.vit_backbone.named_parameters():
                if "gate" in n:
                    p.requires_grad = True

        if self.cfg.freeze_cnn:
            for param in self.cnn_backbone.parameters():
                param.requires_grad = False
        if self.cfg.freeze_vit:
            for param in self.vit_backbone.parameters():
                param.requires_grad = False
                
        # CRITICAL: LoRA/IA3 parameters must remain trainable regardless of freezing
        for m in [self.cnn_backbone, self.vit_backbone]:
            if m is None or isinstance(m, nn.Identity): continue
            for n, p in m.named_parameters():
                if "lora_" in n or "gate" in n:
                    p.requires_grad = True
        
        # Pre-initialize attributes to avoid AttributeError during detection
        self.use_cbam = self.cfg.use_cbam
        self.cbam = None
        self.mixstyle_module = None
        
        if self.use_cbam:
            self._inject_cbam_interleaved()
        

        # Auto-detect dimensions using dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            # CNN Dim
            try:
                feat = self._cnn_features(dummy)
                self.cnn_dim = feat.shape[1]
            except Exception as e:
                LOGGER.warning(f"Error checking CNN dimensions: {e}")
                self.cnn_dim = getattr(self.cnn_backbone, "num_features", 1024)

            # ViT Dim
            try:
                feat = self._vit_features(dummy)
                self.vit_dim = feat.shape[1]
            except Exception as e:
                 LOGGER.warning(f"Error checking ViT dimensions: {e}")
                 self.vit_dim = getattr(self.vit_backbone, "num_features", 0) if self.cfg.vit_model else 0
        
        LOGGER.info(f"HybridBackbone Dimensions Detected -> CNN: {self.cnn_dim}, ViT: {self.vit_dim}")

        # Now instantiate modules dependent on dimensions

            
        if self.cfg.mixstyle:
            self.mixstyle_module = MixStyle(p=self.cfg.mixstyle_p, alpha=self.cfg.mixstyle_alpha)

        # self.token_learner already initialized to None at start
        if self.cfg.token_learner_tokens and self.cfg.vit_model:
            self.token_learner = TokenLearner(self.vit_dim, self.cfg.token_learner_tokens)

        total_dim = self.cnn_dim + self.vit_dim
        self.dha = DynamicHeadAttention(total_dim)
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_dim, self.cfg.fusion_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.cfg.fusion_dim),
        )



    def _inject_cbam_interleaved(self):
        """
        Injects CBAM blocks between stages/blocks of the CNN backbone.
        Supports MobileNetV3 (timm) and ResNet (timm).
        """
        model = self.cnn_backbone
        dummy = torch.zeros(1, 3, 224, 224)
        
        # --- MobileNetV3 Support ---
        if hasattr(model, 'blocks') and isinstance(model.blocks, nn.Sequential):
            LOGGER.info("Detected MobileNetV3-style backbone. Injecting Interleaved CBAM...")
            
            # 1. Run Stem
            try:
                with torch.no_grad():
                    x = model.conv_stem(dummy)
                    x = model.bn1(x)
                    if hasattr(model, 'act1'):
                        x = model.act1(x)
                    elif hasattr(model, 'act'):
                        x = model.act(x)
                    else:
                        LOGGER.warning("No activation found in stem (checked act1, act). Skipping.")
            except Exception as e:
                LOGGER.error(f"Error in Stem Forward: {e}. Available: {dir(model)}")
                raise e
            
            new_blocks = nn.Sequential()
            
            for i, block in enumerate(model.blocks):
                # Run block to get output shape
                with torch.no_grad():
                    x = block(x)
                
                # Add block to new sequential
                new_blocks.add_module(f"block_{i}", block)
                
                # Check for NaNs/Issues? No, just shape.
                channels = x.size(1)
                
                # Create and Add CBAM
                # Use reduction=4 for lighter overhead on small models if desired, but 8 is standard
                cbam = CBAM(channels, reduction_ratio=4) 
                new_blocks.add_module(f"cbam_{i}", cbam)
                
            model.blocks = new_blocks
            LOGGER.info(f"Successfully injected {len(model.blocks)//2} CBAM modules into MobileNetV3 blocks.")
            return

        # --- ResNet Support ---
        # Iterate over layer1, layer2, layer3, layer4
        layers = ['layer1', 'layer2', 'layer3', 'layer4']
        if all(hasattr(model, l) for l in layers):
             LOGGER.info("Detected ResNet-style backbone. Injecting Interleaved CBAM...")
             with torch.no_grad():
                 x = model.conv1(dummy)
                 x = model.bn1(x)
                 x = model.act1(x)
                 x = model.maxpool(x)
             
             for l_name in layers:
                 layer_container = getattr(model, l_name)
                 
                 # It's usually a Sequential of Blocks. We can inject between them or just at end of stage.
                 # User asked "between every two conv layers" -> effectively "between blocks".
                 new_layer = nn.Sequential()
                 
                 for i, block in enumerate(layer_container):
                     with torch.no_grad():
                         x = block(x)
                     
                     new_layer.add_module(f"block_{i}", block)
                     cbam = CBAM(x.size(1), reduction_ratio=8)
                     new_layer.add_module(f"cbam_{i}", cbam)
                     
                 setattr(model, l_name, new_layer)
             
             LOGGER.info("Successfully injected CBAM modules into ResNet layers.")
             return
             
        LOGGER.warning("Could not detect supported structure for Interleaved CBAM. Falling back to single CBAM at end.")
        self.cnn_dims = self._detect_cnn_dims()
        self.cbam = CBAM(self.cnn_dims) # Fallback to old behavior

    def _cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn_backbone.forward_features(x)
        if torch.isnan(feat).any():
            LOGGER.error(f"NaN detected in RAW CNN BACKBONE output! Input stats: mean={x.mean()}, std={x.std()}")
            raise RuntimeError("NaN detected in RAW CNN BACKBONE output!")

        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        
        if self.use_cbam and getattr(self, "cbam", None) is not None and isinstance(feat, torch.Tensor) and feat.dim() == 4:
            feat = self.cbam(feat)
            if torch.isnan(feat).any():
                print("NaN detected in CBAM output!")
                raise RuntimeError("NaN detected in output!")

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
        if not self.cfg.vit_model:
             # Return empty tensor with same batch size and dtype, but 0 features
             return torch.zeros((x.size(0), 0), device=x.device, dtype=x.dtype)

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
        if torch.isnan(cnn_feat).any() or torch.isinf(cnn_feat).any():
             LOGGER.error(f"NaN/Inf detected in CNN Features! min: {cnn_feat.min()}, max: {cnn_feat.max()}")
             raise RuntimeError("NaN/Inf detected in CNN Features!")

        vit_feat = self._vit_features(x)
        if torch.isnan(vit_feat).any() or torch.isinf(vit_feat).any():
             LOGGER.error(f"NaN/Inf detected in ViT Features! min: {vit_feat.min()}, max: {vit_feat.max()}")
             raise RuntimeError("NaN/Inf detected in ViT Features!")

        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        
        refined = self.dha(combined)
        if torch.isnan(refined).any():
             LOGGER.error("NaN detected in DHA (Attention) Output!")
             raise RuntimeError("NaN detected in DHA (Attention) Output!")
             
        fused = self.fusion_fc(refined)
        if torch.isnan(fused).any():
             LOGGER.error("NaN detected in Fusion FC Output!")
             raise RuntimeError("NaN detected in Fusion FC Output!")
             
        return fused


if __name__ == "__main__":
    try:
        cfg = BackboneConfig(
            vit_model="dinov2_base14",
            cnn_lora_rank=8,
            vit_lora_rank=8,
            token_merging_ratio=0.5,
            vit_ia3=True,
        )
        if torch.cuda.is_available():
            model = HybridBackbone(cfg).cuda()
            dummy = torch.randn(2, 3, 224, 224).cuda()
            out = model(dummy)
            # print("Output Shape:", out.shape)
        LOGGER.info("Backbone module loaded successfully.")
    except Exception as e:
        LOGGER.warning(f"Backbone smoke test failed: {e}")


