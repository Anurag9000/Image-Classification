import torch
import os
from pipeline.backbone import HybridBackbone, BackboneConfig
from pipeline.train_arcface import AdaFace
import logging

def create_dummy_ckpt():
    print("Creating dummy checkpoint...")
    
    # 1. Config matching test_distill.yaml's teacher expectations
    # Teacher backbone is MobileNetV3
    # Explicitly disable ViT to avoid dimension mismatch (192 vs 224) and default instantiation
    bb_cfg = BackboneConfig(cnn_model="mobilenetv3_large_100", vit_model=None, pretrained=False)
    backbone = HybridBackbone(bb_cfg)
    
    # Robustly get fusion dim
    backbone.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = backbone(dummy)
    fusion_dim = out.shape[1]
    print(f"Backbone dim determined via forward pass: {fusion_dim}")
    
    # Head
    # Match dataset classes (organic, other, paper, plastic = 4)
    num_classes = 4 
    head = AdaFace(embedding_size=fusion_dim, num_classes=num_classes)
    
    # 2. Save State
    state = {
        "model_state_dict": backbone.state_dict(),
        "head_state_dict": head.state_dict()
    }
    
    os.makedirs("./snapshots_test", exist_ok=True)
    save_path = "./snapshots_test/best_model.pth"
    torch.save(state, save_path)
    print(f"Saved dummy checkpoint to {save_path} with keys: {list(state.keys())}")

if __name__ == "__main__":
    create_dummy_ckpt()
