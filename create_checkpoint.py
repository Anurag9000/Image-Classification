import torch
import os
from pipeline.backbone import HybridBackbone, BackboneConfig
from pipeline.train_arcface import AdaFace
import logging

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def create_dummy_ckpt():
    LOGGER.info("Creating dummy checkpoint...")
    
    # 1. Load Real Config
    config_path = "configs/config_edge.yaml"
    if not os.path.exists(config_path):
        LOGGER.warning(f"Config not found: {config_path}. Using hardcoded defaults.")
        # Fallback only if config is missing
        cnn_model = "mobilenetv3_large_100"
        num_classes = 4
    else:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # We need the TEACHER config (Phase 1/2) if we are simulating the teacher.
        # configs/config_edge.yaml top-level 'backbone' is the teacher ID (e.g. resnet50).
        # configs/config_edge.yaml 'distill' section defines the student.
        # But wait, create_checkpoint.py says "Match test_distill.yaml teacher expectations".
        # test_distill.yaml usually expects the teacher to be loaded.
        
        # Let's inspect what config values we actually need.
        # Extract values from config or defaults
        # We try to find the 'student' or 'distill' backbone config usually, but falling back to global backbone is safer if distill is not present.
        
        # Priority 1: Distill backbone (Student)
        if 'distill' in cfg and 'backbone' in cfg['distill']:
             cnn_model = cfg['distill']['backbone'].get('cnn_model', 'mobilenetv3_large_100')
             vit_model = cfg['distill']['backbone'].get('vit_model', None)
        # Priority 2: Global backbone (Teacher)
        elif 'backbone' in cfg:
             cnn_model = cfg['backbone'].get('cnn_model', 'mobilenetv3_large_100')
             vit_model = cfg['backbone'].get('vit_model', None)
        else:
             cnn_model = "mobilenetv3_large_100"
             vit_model = None
             
        num_classes = cfg.get('num_classes', 4)
        LOGGER.info(f"Loaded config: CNN={cnn_model}, ViT={vit_model}, Classes={num_classes}")

    
    bb_cfg = BackboneConfig(cnn_model=cnn_model, vit_model=vit_model, pretrained=False)
    backbone = HybridBackbone(bb_cfg)
    
    # Robustly get fusion dim
    backbone.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = backbone(dummy)
    fusion_dim = out.shape[1]
    LOGGER.info(f"Backbone dim determined via forward pass: {fusion_dim}")
    
    # Head
    head = AdaFace(embedding_size=fusion_dim, num_classes=num_classes)
    
    # 2. Save State
    state = {
        "model_state_dict": backbone.state_dict(),
        "head_state_dict": head.state_dict()
    }
    
    os.makedirs("./snapshots_test", exist_ok=True)
    save_path = "./snapshots_test/best_model.pth"
    torch.save(state, save_path)
    LOGGER.info(f"Saved dummy checkpoint to {save_path} with keys: {list(state.keys())}")

if __name__ == "__main__":
    create_dummy_ckpt()
