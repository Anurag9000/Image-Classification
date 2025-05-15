# backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# ------------------------------
# Convolutional Block Attention Module (CBAM)
# ------------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Channel attention
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        channel_att = torch.sigmoid(self.fc(avg_pool) + self.fc(max_pool)).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x

# ------------------------------
# Dynamic Head Attention Fusion (DHA)
# ------------------------------
class DynamicHeadAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(DynamicHeadAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, vit_feat):
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        attn_weights = self.fc(combined)
        refined = combined * attn_weights
        return refined

# ------------------------------
# Hybrid Backbone: ConvNeXt-V2 + SwinV2 + CBAM + DHA Fusion
# ------------------------------
class HybridBackbone(nn.Module):
    def __init__(self, use_cbam=True):
        super(HybridBackbone, self).__init__()

        # ConvNeXt-V2 backbone (ImageNet-1K pretrained)
        self.cnn_backbone = create_model('convnextv2_base', pretrained=True)
        self.cnn_backbone.head = nn.Identity()

        # SwinV2 backbone (ImageNet-1K pretrained)
        self.vit_backbone = create_model('swinv2_base_window12_192_22k', pretrained=True)
        self.vit_backbone.head = nn.Identity()

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(1024)

        # Dynamic Head Attention for feature fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(1024 + 1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024)
        )
        self.dha = DynamicHeadAttention(1024 * 2)

    def forward(self, x):
        # CNN Features
        cnn_feat = self.cnn_backbone.forward_features(x)
        if self.use_cbam:
            cnn_feat_map = self.cnn_backbone.stages[-1](self.cnn_backbone.stages[-2](x))
            cnn_feat_map = self.cbam(cnn_feat_map)
            cnn_feat = F.adaptive_avg_pool2d(cnn_feat_map, 1).view(cnn_feat_map.size(0), -1)

        # ViT Features
        vit_feat = self.vit_backbone.forward_features(x)

        # Dynamic Head Attention Fusion
        refined_feat = self.dha(cnn_feat, vit_feat)
        fused = self.fusion_fc(refined_feat)
        return fused


if __name__ == "__main__":
    model = HybridBackbone(use_cbam=True).cuda()
    dummy = torch.randn(2, 3, 224, 224).cuda()
    out = model(dummy)
    print(f"Output Shape: {out.shape}")  # Should be (B, 1024)
