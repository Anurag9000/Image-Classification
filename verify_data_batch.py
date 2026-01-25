import torch
import os
import yaml
from pipeline.files_dataset import create_data_loader
import logging
import sys

import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def verify_data(config_path="configs/config_edge.yaml"):
    LOGGER.info("=== Verifying Data Integrity ===")
    
    # Load config
    LOGGER.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    dataset_cfg = cfg["dataset"]
    LOGGER.info(f"Loading from: {dataset_cfg['root_dirs']}")
    LOGGER.info(f"JSON Path: {dataset_cfg['json_path']}")
    
    # Create Ioader
    batch_size = dataset_cfg.get('batch_size', 64)
    LOGGER.info(f"Using batch size: {batch_size}")
    
    try:
        train_loader, _, _ = create_data_loader(
            root_dirs=dataset_cfg["root_dirs"],
            batch_size=batch_size,
            num_workers=0, # Debug on main thread
            val_split=0.1,
            test_split=0.1,
            json_path=dataset_cfg["json_path"]
        )
    except Exception as e:
        LOGGER.error(f"Loader Creation Failed: {e}")
        return

    LOGGER.info(f"Loader Length (Batches): {len(train_loader)}")
    
    # Fetch one batch
    try:
        images, labels = next(iter(train_loader))
    except Exception as e:
        LOGGER.error(f"Batch Fetch Failed: {e}")
        return
        
    LOGGER.info(f"Batch Shape: Images={images.shape}, Labels={labels.shape}")
    LOGGER.info(f"Image Range: Min={images.min():.4f}, Max={images.max():.4f}")
    LOGGER.info(f"Image Mean: {images.mean():.4f}, Std: {images.std():.4f}")
    
    LOGGER.info(f"Unique Labels in Batch: {torch.unique(labels).tolist()}")
    
    if torch.isnan(images).any():
        LOGGER.critical("NaNs found in input images!")
    else:
        LOGGER.info("Input images are clean (No NaNs).")
        
    if labels.max() >= cfg["num_classes"]:
        LOGGER.critical(f"Label {labels.max()} exceeds num_classes {cfg['num_classes']}")
    else:
        LOGGER.info(f"Labels are within range [0, {cfg['num_classes']-1}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_edge.yaml")
    args = parser.parse_args()
    verify_data(args.config)
