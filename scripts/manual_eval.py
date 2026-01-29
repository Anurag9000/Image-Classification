
import torch
import os
import argparse
from pipeline.train_arcface import ArcFaceConfig, ArcFaceTrainer
from pipeline.files_dataset import create_data_loader
from sklearn.metrics import accuracy_score

def manual_eval():
    print("Loading Final Backbone...")
    snapshot_dir = "./snapshots_tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Config from yaml to get dataset paths
    # Quick hack: explicit paths matching config_tiny.yaml
    root_dirs = ["./data/Dataset_Final"]
    json_path = "./data/Dataset_Final/dataset_metadata.json"
    
    # Create Test Loader
    _, _, test_loader = create_data_loader(
        root_dirs=root_dirs,
        json_path=json_path,
        batch_size=64,
        num_workers=4,
        val_split=0.1,
        test_split=0.1,
        augment_online=False
    )
    
    # Init Model Structure
    # MobileNetV3 Large
    cfg = ArcFaceConfig(
        num_classes=4,
        backbone={"cnn_model": "mobilenetv3_large_100", "pretrained": True, "vit_model": None, "use_cbam": False},
        image_size=224
    )
    trainer = ArcFaceTrainer(None, None, cfg)
    
    # Load Backbone (Final)
    backbone_path = os.path.join(snapshot_dir, "backbone_final.pth")
    print(f"Loading Backbone from: {backbone_path}")
    if os.path.exists(backbone_path):
        state = torch.load(backbone_path, map_location=device)
        trainer.backbone.load_state_dict(state)
    else:
        print("Backbone file not found!")
        
    # Load Head (Final)
    head_path = os.path.join(snapshot_dir, "head", "head_final.pth")
    print(f"Loading Head from: {head_path}")
    if os.path.exists(head_path):
        state = torch.load(head_path, map_location=device)
        trainer.head.load_state_dict(state)
    else:
        print("Head file not found!")
            
    trainer.backbone.eval()
    trainer.head.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            feats = trainer.backbone(images)
            logits = trainer.head(feats, labels=None) # Inference mode
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    print(f"MANUAL EVAL ACCURACY: {acc*100:.2f}%")

if __name__ == "__main__":
    manual_eval()
