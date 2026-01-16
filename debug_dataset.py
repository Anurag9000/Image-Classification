
import os
import sys
import yaml
import torch
import cv2
import pandas as pd
from pipeline.files_dataset import CombinedFilesDataset

def debug_dataset(config_path):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    root_dirs = cfg['dataset']['root_dirs']
    print(f"Dataset Roots: {root_dirs}")

    # Load dataset without transforms to see raw paths
    dataset = CombinedFilesDataset(root_dirs, split='train', transform=None)
    
    print(f"\nTotal Samples Found: {len(dataset)}")
    print(f"Number of Classes: {len(dataset.class_names)}")
    print(f"Class Names: {dataset.class_names}")

    print("\n--- Sample Verification (First 5) ---")
    for i in range(5):
        img_path, label_idx = dataset.samples[i]
        label_name = dataset.class_names[label_idx]
        print(f"Index {i}:")
        print(f"  Image: {img_path}")
        print(f"  Label Index: {label_idx} -> Name: {label_name}")
        
        if not os.path.exists(img_path):
            print("  [ERROR] Image file does not exist!")
        else:
            print("  [OK] Image file exists.")

    print("\n--- Distribution Check ---")
    label_counts = {}
    for _, label_idx in dataset.samples:
        label_name = dataset.class_names[label_idx]
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
    
    for name, count in label_counts.items():
        print(f"  {name}: {count} images")

if __name__ == "__main__":
    config_file = "configs/garbage_v3.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    debug_dataset(config_file)
