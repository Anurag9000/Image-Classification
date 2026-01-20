
import os
import sys
import yaml
import torch
import cv2
import pandas as pd
from pipeline.files_dataset import CombinedFilesDataset

import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def debug_dataset(config_path):
    LOGGER.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    root_dirs = cfg['dataset']['root_dirs']
    LOGGER.info(f"Dataset Roots: {root_dirs}")

    # Load dataset without transforms to see raw paths
    dataset = CombinedFilesDataset(root_dirs, split='train', transform=None)
    
    LOGGER.info(f"\nTotal Samples Found: {len(dataset)}")
    LOGGER.info(f"Number of Classes: {len(dataset.class_names)}")
    LOGGER.info(f"Class Names: {dataset.class_names}")

    LOGGER.info("\n--- Sample Verification (First 5) ---")
    for i in range(5):
        if i >= len(dataset):
             break
        img_path, label_idx = dataset.samples[i]
        label_name = dataset.class_names[label_idx]
        LOGGER.info(f"Index {i}:")
        LOGGER.info(f"  Image: {img_path}")
        LOGGER.info(f"  Label Index: {label_idx} -> Name: {label_name}")
        
        if not os.path.exists(img_path):
            LOGGER.error("  [ERROR] Image file does not exist!")
        else:
            LOGGER.info("  [OK] Image file exists.")

    LOGGER.info("\n--- Distribution Check ---")
    label_counts = {}
    for _, label_idx in dataset.samples:
        label_name = dataset.class_names[label_idx]
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
    
    for name, count in label_counts.items():
        LOGGER.info(f"  {name}: {count} images")

if __name__ == "__main__":
    config_file = "configs/garbage_edge.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    debug_dataset(config_file)
