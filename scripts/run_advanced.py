#!/usr/bin/env python3
import os
import yaml
import logging
from pipeline.train_supcon import SupConPretrainer as SupConTrainer, SupConConfig
from pipeline.train_arcface import ArcFaceTrainer, ArcFaceConfig
from pipeline.files_dataset import create_data_loader

def main():
    # Load Config
    with open("configs/config_advanced.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
        
    print("Loaded Advanced Config (SupCon + ArcFace).")
    
    # -------------------------------------------------------------------------
    # Parsing Config Sections
    # -------------------------------------------------------------------------
    dataset_cfg = cfg_dict.get('dataset', {})
    backbone_cfg = cfg_dict.get('backbone', {})
    supcon_cfg_dict = cfg_dict.get('supcon', {})
    arcface_cfg_dict = cfg_dict.get('arcface', {})
    
    # -------------------------------------------------------------------------
    # Shared Data Loaders
    # -------------------------------------------------------------------------
    # SupCon usually uses a specialized TwoCropTransform, but our current SupConTrainer
    # handles generic loaders and applies mixup/views internally or expects the loader to return them.
    # Let's check SupConTrainer logic: It usually needs Multi-View sampling.
    # If our SupConTrainer relies on 'augment_online' it might use standard augs.
    # Standard SimCLR needs: X -> (X1, X2).
    # If our files_dataset doesn't support SimCLR transform, SupCon might fail or be weak.
    # However, existing repo implies SupConTrainer works with the current loader.
    # We will use the standard loader.
    
    train_loader, val_loader, _ = create_data_loader(
        root_dirs=dataset_cfg.get('root_dirs'),
        batch_size=dataset_cfg.get('batch_size', 64),
        num_workers=dataset_cfg.get('num_workers', 4),
        val_split=dataset_cfg.get('val_split', 0.1),
        test_split=dataset_cfg.get('test_split', 0.1),
        json_path=dataset_cfg.get('json_path'),
        augment_online=dataset_cfg.get('augment_online', True)
    )
    
    print("Data Loaders Created.")
    
    # -------------------------------------------------------------------------
    # Phase 1: SupCon (Supervised Contrastive Pre-training)
    # -------------------------------------------------------------------------
    if supcon_cfg_dict.get('enabled', False):
        print("\n=== Phase 1: Supervised Contrastive Pre-training ===")
        from pipeline.train_supcon import create_supcon_loader
        
        # SupCon needs Multi-View Loader
        # We pass dataset_cfg.get('root_dirs')[0] if available, else json_path dir
        roots = dataset_cfg.get('root_dirs', ["./data"])
        root_dir = roots[0] if roots else "./data"
        
        sup_train_loader, sup_val_loader = create_supcon_loader(
            batch_size=int(supcon_cfg_dict.get('batch_size', 64)),
            root=root_dir, # Use the actual data root
            num_workers=dataset_cfg.get('num_workers', 4),
            json_path=dataset_cfg.get('json_path'),
            num_views=4  # CRITICAL: Must match SupConConfig.num_views
        )
        
        
        # Prepare Config
        s_cfg = SupConConfig(
            backbone=backbone_cfg,
            steps=int(supcon_cfg_dict.get('steps', 482000)),
            lr=float(supcon_cfg_dict.get('lr', 1e-3)),
            snapshot_path=supcon_cfg_dict.get('snapshot_path', "./snapshots_advanced/supcon_final.pth"),
            use_sam=supcon_cfg_dict.get('use_sam', False),
            rho=supcon_cfg_dict.get('rho', 0.05),
            use_amp=supcon_cfg_dict.get('use_amp', True),
            image_size=224,
            resume_from=supcon_cfg_dict.get('resume_from', None)
        )
        
        trainer = SupConTrainer(sup_train_loader, sup_val_loader, s_cfg)
        trainer.train()
        
        # Verify Snapshot
        if os.path.exists(s_cfg.snapshot_path):
            print(f"SupCon Phase Complete. Saved to {s_cfg.snapshot_path}")
        else:
            print("WARNING: SupCon Snapshot not found. Phase 2 might fail or strict load will error.")
    else:
        print("Skipping SupCon Phase (Disabled in Config).")

    # -------------------------------------------------------------------------
    # Phase 2: ArcFace (Classification Fine-Tuning)
    # -------------------------------------------------------------------------
    if arcface_cfg_dict.get('enabled', True):
        print("\n=== Phase 2: ArcFace Fine-Tuning (ULMFiT) ===")
        
        # Prepare Config
        a_cfg = ArcFaceConfig(
            num_classes=cfg_dict.get('num_classes', 4),
            backbone=backbone_cfg,
            epochs=arcface_cfg_dict.get('epochs', 20),
            lr=float(arcface_cfg_dict.get('lr', 1e-4)),
            snapshot_dir=arcface_cfg_dict.get('snapshot_dir', "./snapshots_advanced"),
            
            # ULMFiT Stats
            use_ulmfit=arcface_cfg_dict.get('use_ulmfit', False),
            gradual_unfreezing=arcface_cfg_dict.get('gradual_unfreezing', False),
            unfreeze_epoch=arcface_cfg_dict.get('unfreeze_epoch', 2),
            discriminative_lr_decay=arcface_cfg_dict.get('discriminative_lr_decay', 2.6),
            val_limit_batches=arcface_cfg_dict.get('val_limit_batches', None),
            early_stopping_patience=arcface_cfg_dict.get('early_stopping_patience', 5),
            
            # Others
            use_sam=arcface_cfg_dict.get('use_sam', False),
            use_amp=arcface_cfg_dict.get('use_amp', True),
            rho=arcface_cfg_dict.get('rho', 0.05),
            
            # Link to Phase 1
            supcon_snapshot=supcon_cfg_dict.get('snapshot_path', "./snapshots_advanced/supcon_final.pth") if supcon_cfg_dict.get('enabled') else None
        )
        
        trainer = ArcFaceTrainer(train_loader, val_loader, a_cfg)
        trainer.train()
        print("ArcFace Phase Complete.")
    
    print("\nAdvanced Pipeline Finished Successfully.")

if __name__ == "__main__":
    
    # Setup Logging
    os.makedirs("./logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("./logs/training.log"),
            logging.StreamHandler()
        ]
    )
    main()
