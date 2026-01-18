import torch
import os
import yaml
from pipeline.files_dataset import create_garbage_loader
import logging
import sys

def verify_data():
    print("=== Verifying Data Integrity ===")
    
    # Load config
    with open("configs/garbage_edge.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    dataset_cfg = cfg["dataset"]
    print(f"Loading from: {dataset_cfg['root_dirs']}")
    print(f"JSON Path: {dataset_cfg['json_path']}")
    
    # Create Ioader
    try:
        train_loader, _, _ = create_garbage_loader(
            root_dirs=dataset_cfg["root_dirs"],
            batch_size=64,
            num_workers=0, # Debug on main thread
            val_split=0.1,
            test_split=0.1,
            json_path=dataset_cfg["json_path"]
        )
    except Exception as e:
        print(f"Loader Creation Failed: {e}")
        return

    print(f"Loader Length (Batches): {len(train_loader)}")
    
    # Fetch one batch
    try:
        images, labels = next(iter(train_loader))
    except Exception as e:
        print(f"Batch Fetch Failed: {e}")
        return
        
    print(f"Batch Shape: Images={images.shape}, Labels={labels.shape}")
    print(f"Image Range: Min={images.min():.4f}, Max={images.max():.4f}")
    print(f"Image Mean: {images.mean():.4f}, Std: {images.std():.4f}")
    
    print(f"Unique Labels in Batch: {torch.unique(labels).tolist()}")
    
    if torch.isnan(images).any():
        print("CRITICAL: NaNs found in input images!")
    else:
        print("SUCCESS: Input images are clean (No NaNs).")
        
    if labels.max() >= cfg["num_classes"]:
        print(f"CRITICAL: Label {labels.max()} exceeds num_classes {cfg['num_classes']}")
    else:
        print(f"SUCCESS: Labels are within range [0, {cfg['num_classes']-1}]")

if __name__ == "__main__":
    verify_data()
