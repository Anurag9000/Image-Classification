#!/usr/bin/env python3
import os
import yaml
import logging
from pipeline.fine_tune_distill import FineTuneDistillTrainer, DistillConfig, create_distill_loader

def main():
    # Load Config
    with open("configs/config_advanced.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
        
    print("Loaded Advanced Config.")
    
    # Create Config Object
    # Flatten config dict for dataclass
    # We need to map nested dicts to fields manually or use **kwargs if class supports it
    # DistillConfig has 'backbone', 'teacher_backbone_config', 'distill', etc.
    
    # Merge 'distill' section into main config
    distill_cfg = cfg_dict.pop('distill', {})
    dataset_cfg = cfg_dict.pop('dataset', {})
    backbone_cfg = cfg_dict.pop('backbone', {})
    teacher_cfg = cfg_dict.pop('teacher_backbone_config', {})
    
    # Init Config
    distill_cfg.pop('enabled', None)  # Remove 'enabled' key
    distill_cfg.pop('snapshot_path', None) # Remove if present
    
    config = DistillConfig(
        num_classes=cfg_dict.get('num_classes', 4),
        backbone=backbone_cfg,
        teacher_backbone_config=teacher_cfg,
        **distill_cfg # Unpack distill settings (epochs, lr, etc.)
    )
    
    # Data Loaders
    train_loader, val_loader = create_distill_loader(
        root=dataset_cfg.get('root_dirs'),
        batch_size=dataset_cfg.get('batch_size', 32),
        num_workers=dataset_cfg.get('num_workers', 4),
        val_split=dataset_cfg.get('val_split', 0.1),
        json_path=dataset_cfg.get('json_path'),
        augment_online=dataset_cfg.get('augment_online', True)
    )
    
    print("Data Loaders Created.")
    
    # Init Trainer
    trainer = FineTuneDistillTrainer(train_loader, val_loader, config)
    
    # Train
    print("Starting Advanced Training (EfficientNet + CBAM + ULMFiT)...")
    trainer.train()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
