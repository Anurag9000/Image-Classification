import torch
import torch.nn as nn
from pipeline.backbone import (
    BackboneConfig, 
    HybridBackbone, 
    TokenLearner, 
    MixStyle, 
    CBAM, 
    DynamicHeadAttention
)

def test_token_learner():
    tl = TokenLearner(embed_dim=128, num_output_tokens=8)
    x = torch.randn(2, 64, 128) # (B, N, C)
    out = tl(x)
    assert out.shape == (2, 8, 128)

def test_mixstyle():
    ms = MixStyle(p=1.0, alpha=0.1)
    ms.train()
    x = torch.randn(4, 16, 8, 8)
    out = ms(x)
    assert out.shape == x.shape

def test_cbam():
    cbam = CBAM(channels=16)
    x = torch.randn(2, 16, 8, 8)
    out = cbam(x)
    assert out.shape == x.shape

def test_dha():
    dha = DynamicHeadAttention(input_dim=128)
    x = torch.randn(2, 128)
    out = dha(x)
    assert out.shape == (2, 128)

def test_hybrid_backbone():
    # Using tiny models for fast unit testing
    cfg = BackboneConfig(
        cnn_model="resnet18",
        vit_model="vit_tiny_patch16_224",
        pretrained=False,
        fusion_dim=128,
        use_cbam=True,
        mixstyle=True
    )
    model = HybridBackbone(cfg)
    model.eval()
    
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    
    # Check output dim
    assert out.shape == (1, 128)
    
    # Test with lora
    cfg_lora = BackboneConfig(
        cnn_model="resnet18",
        vit_model="vit_tiny_patch16_224",
        pretrained=False,
        lora_rank=4,
        fusion_dim=128
    )
    model_lora = HybridBackbone(cfg_lora)
    assert any("lora_A" in name for name, _ in model_lora.named_parameters())
