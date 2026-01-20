from __future__ import annotations

import argparse
import logging
import os
import torch
from typing import List

import yaml
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


from .evaluate import EvaluationConfig, Evaluator
from .fine_tune_distill import DistillConfig, FineTuneDistillTrainer, create_distill_loader
from .train_arcface import ArcFaceConfig, ArcFaceTrainer, create_dataloader as create_arcface_loader
from .train_supcon import SupConConfig, SupConPretrainer, create_supcon_loader
from pipeline.files_dataset import create_garbage_loader
from utils import setup_logger

LOGGER = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)





def run_supcon_phase(full_cfg: dict) -> None:
    cfg = full_cfg.get("supcon", {})
    if not cfg.get("enabled", True):
        LOGGER.info("SupCon phase disabled, skipping.")
        return

    LOGGER.info("===> Starting SupCon Pretraining Phase")
    
    # Extract json_path from global dataset config
    json_path = None
    if "dataset" in full_cfg and "json_path" in full_cfg["dataset"]:
        json_path = full_cfg["dataset"]["json_path"]

    # Use global backbone if not in supcon config
    backbone_cfg = cfg.get("backbone", full_cfg.get("backbone", {}))

    # Fix: Ensure num_views is consistent between Loader and Trainer
    num_views = cfg.get("num_views", 2)

    loader, val_loader = create_supcon_loader(
        batch_size=cfg.get("batch_size", 16),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations"),
        root=cfg.get("data_root", "./data"),
        num_workers=cfg.get("num_workers", 0),
        json_path=json_path,
        num_views=num_views 
    )
    supcon_cfg = SupConConfig(
        temperature=float(cfg.get("temperature", 0.07)),
        num_views=int(num_views),
        lr=float(cfg.get("lr", 1e-3)),
        steps=int(cfg.get("steps", 200)),
        ema_decay=float(cfg.get("ema_decay", 0.9995)) if cfg.get("ema_decay", 0.9995) is not None else None,
        backbone=backbone_cfg,
        image_size=int(cfg.get("image_size", 224)),
        augmentations=cfg.get("augmentations", {}),
        num_workers=int(cfg.get("num_workers", 0)),
        max_steps=int(cfg.get("max_steps", 0)) if cfg.get("max_steps") else None,
        snapshot_path=cfg.get("snapshot_path", "./snapshots/supcon_final.pth"),
        early_stopping_patience=int(cfg.get("early_stopping_patience", 10)),
    )
    SupConPretrainer(loader, val_loader, supcon_cfg).train()


def run_arcface_phase(cfg: dict) -> None:
    LOGGER.info("===> Starting ArcFace Training Phase")
    
    if "dataset" in cfg and "root_dirs" in cfg["dataset"]:
        LOGGER.info(f"Using CombinedFilesDataset with roots: {cfg['dataset']['root_dirs']}")
        train_loader, val_loader, test_loader = create_garbage_loader(
            root_dirs=cfg["dataset"]["root_dirs"],
            batch_size=cfg["dataset"].get("batch_size", 32),
            num_workers=cfg["dataset"].get("num_workers", 4),
            val_split=cfg["dataset"].get("val_split", 0.0),
            test_split=cfg["dataset"].get("test_split", 0.0), # Add test split
            json_path=cfg["dataset"].get("json_path", None),
            augment_online=cfg["dataset"].get("augment_online", True)
        )
    else:
        # Legacy/Standard Loader (updates might be needed if this path is used, but for now keep as is or unpack)
        # Note: create_arcface_loader returns 2 items in current code. 
        # If we are strictly using the new path, this is fine. 
        # But for safety, valid/test might be mixed or ignored here.
        train_loader, val_loader = create_arcface_loader(
            batch_size=cfg.get("batch_size", 32),
            augment=cfg.get("augment", True),
            image_size=cfg.get("image_size", 224),
            augmentations=cfg.get("augmentations"),
            root=cfg.get("data_root", "./data"),
            num_workers=cfg.get("num_workers", 0),
            val_split=cfg.get("val_split", 0.1),
        )
        test_loader = None

    # Merge global cfg (for num_classes/backbone) with arcface specific cfg
    # Priority: arcface_cfg > global_cfg > defaults
    arcface_specific = cfg.get("arcface", {})
    
    # Helper to get from either, preferring specific
    def get_cfg(key, default):
        return arcface_specific.get(key, cfg.get(key, default))

    arcface_cfg = ArcFaceConfig(
        num_classes=int(get_cfg("num_classes", 100)),
        lr=float(get_cfg("lr", 1e-5)),
        gamma=float(get_cfg("gamma", 2.0)),
        smoothing=float(get_cfg("smoothing", 0.1)),
        epochs=int(get_cfg("epochs", 30)),
        mix_method=get_cfg("mix_method", "mixup"),
        use_curricularface=get_cfg("use_curricularface", True),
        use_evidential=get_cfg("use_evidential", False),
        ema_decay=float(get_cfg("ema_decay", 0.9995)) if get_cfg("ema_decay", 0.9995) is not None else None,
        compile_model=get_cfg("compile", False),
        backbone=get_cfg("backbone", {}),
        image_size=int(get_cfg("image_size", 224)),
        augmentations=get_cfg("augmentations", {}),
        use_manifold_mixup=get_cfg("use_manifold_mixup", False),
        manifold_mixup_alpha=float(get_cfg("manifold_mixup_alpha", 2.0)),
        manifold_mixup_weight=float(get_cfg("manifold_mixup_weight", 0.5)),
        max_steps=int(get_cfg("max_steps", 0)) if get_cfg("max_steps", None) else None,
        snapshot_dir=get_cfg("snapshot_dir", "./snapshots"),
        log_csv=get_cfg("log_csv", "./logs/arcface_metrics.csv"),
        early_stopping_patience=int(get_cfg("early_stopping_patience", 5)),
        val_split=float(get_cfg("val_split", 0.1)),
        use_amp=get_cfg("use_amp", True),
        # Pass SupCon Snapshot Path - Prefer BEST model
        supcon_snapshot=cfg.get("supcon_snapshot", "./snapshots/supcon_final_best.pth"), 
    )
    
    trainer = ArcFaceTrainer(train_loader, val_loader, arcface_cfg)
    trainer.train()

    # --- Final Evaluation on Test Set ---
    if test_loader:
        LOGGER.info("===> Running Final Evaluation on Test Set (Best Model)")
        # Load best model
        best_model_path = os.path.join(arcface_cfg.snapshot_dir, "best_model.pth")
        if os.path.exists(best_model_path):
             checkpoint = torch.load(best_model_path, map_location=trainer.device)
             if "model_state_dict" in checkpoint:
                 trainer.backbone.load_state_dict(checkpoint["model_state_dict"], strict=False)
             elif "backbone" in checkpoint:
                 trainer.backbone.load_state_dict(checkpoint["backbone"], strict=False)
                 
             # Head usually needs loading too for ArcFace/Loss-based metrics, 
             # but for pure inference usually we checking feature similarity or using head as classifier?
             # For this pipeline, 'head' is part of training (AdaFace). 
             # But for pure classification accuracy, we often use the backbone features or the head logits.
             # The Trainer._validate uses trainer.head() so we should load it.
             if "head_state_dict" in checkpoint:
                 trainer.head.load_state_dict(checkpoint["head_state_dict"], strict=False)
             elif "head" in checkpoint:
                 trainer.head.load_state_dict(checkpoint["head"], strict=False)
                 
        # Re-use validation logic but with test_loader
        trainer.val_loader = test_loader
        test_loss, test_acc, test_f1 = trainer._validate()
        
        # Calculate F1 as well (Validation loop updated previously didn't do F1 in _validate return, but calculates internally in epoch loop)
        # Let's do a quick manual run to get F1
        trainer.backbone.eval()
        trainer.head.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(trainer.device)
                labels = labels.to(trainer.device)
                feats = trainer.backbone(images)
                logits = trainer.head(feats, labels=None)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                 
        from sklearn.metrics import accuracy_score, f1_score
        final_acc = accuracy_score(all_labels, all_preds)
        final_f1 = f1_score(all_labels, all_preds, average='macro')
        
        LOGGER.info(f"FINAL TEST RESULTS :: Accuracy: {final_acc:.4f} | F1 Score: {final_f1:.4f}")


def run_distill_phase(full_cfg: dict) -> None: # Received full config
    cfg = full_cfg.get("distill", {})
    if not cfg.get("enabled", True):
        LOGGER.info("Distillation phase disabled, skipping.")
        return

    LOGGER.info("===> Starting Fine-Tune + Distillation Phase")
    # Extract json_path from global dataset config
    json_path = None
    if "dataset" in full_cfg and "json_path" in full_cfg["dataset"]:
        json_path = full_cfg["dataset"]["json_path"]

    train_loader, val_loader = create_distill_loader(
        batch_size=int(cfg.get("batch_size", 32)),
        image_size=int(cfg.get("image_size", 224)),
        augmentations=cfg.get("augmentations"),
        root=cfg.get("data_root", "./data"),
        num_workers=int(cfg.get("num_workers", 0)),
        val_split=float(cfg.get("val_split", 0.1)),
        json_path=json_path,
        augment_online=full_cfg.get("dataset", {}).get("augment_online", True),
    )
    distill_cfg = DistillConfig(
        num_classes=int(cfg.get("num_classes", 100)),
        distill_weight=float(cfg.get("distill_weight", 0.3)),
        lr=float(cfg.get("lr", 5e-6)),
        mix_method=cfg.get("mix_method", "mixup"),
        epochs=int(cfg.get("epochs", 10)),
        ema_decay=float(cfg.get("ema_decay", 0.9995)) if cfg.get("ema_decay", 0.9995) is not None else None,
        teacher_backbone_path=cfg.get("teacher_backbone_path"),
        teacher_head_path=cfg.get("teacher_head_path"),
        backbone=cfg.get("backbone", {}), # Student config
        image_size=int(cfg.get("image_size", 224)),
        augmentations=cfg.get("augmentations", {}),
        use_manifold_mixup=cfg.get("use_manifold_mixup", False),
        manifold_mixup_alpha=float(cfg.get("manifold_mixup_alpha", 2.0)),
        manifold_mixup_weight=float(cfg.get("manifold_mixup_weight", 0.5)),
        use_ema_teacher=cfg.get("use_ema_teacher", True),
        num_workers=int(cfg.get("num_workers", 0)),
        max_steps=int(cfg.get("max_steps", 0)) if cfg.get("max_steps") else None,
        snapshot_dir=cfg.get("snapshot_dir", "./snapshots_distill"),
        log_csv=cfg.get("log_csv", "./logs/distill_metrics.csv"),
        early_stopping_patience=int(cfg.get("early_stopping_patience", 5)),
        val_split=float(cfg.get("val_split", 0.1)),
        
        # Inject correct teacher config from global config
        teacher_backbone_config=full_cfg.get("backbone", {})
    )
    # Filter args to remove unexpected validation issues if any, but DistillConfig needs update first
    FineTuneDistillTrainer(train_loader, val_loader, distill_cfg).train()


def run_evaluation_phase(full_cfg: dict) -> None:
    cfg = full_cfg.get("evaluate", {}) # Still get evaluate section for specific params
    LOGGER.info("===> Starting Evaluation Phase")
    
    if "dataset" in full_cfg:
        LOGGER.info(f"Using CombinedFilesDataset/JsonDataset for Evaluation")
        _, _, loader = create_garbage_loader(
            root_dirs=full_cfg["dataset"]["root_dirs"],
            batch_size=cfg.get("batch_size", 32),
            num_workers=cfg.get("num_workers", 4),
            val_split=full_cfg["dataset"].get("val_split", 0.1),
            test_split=full_cfg["dataset"].get("test_split", 0.1),
            json_path=full_cfg["dataset"].get("json_path", None)
        )
    else:
        LOGGER.error("No global dataset config found. Legacy loader removed as it was broken.")
        return

    eval_cfg = EvaluationConfig(
        result_dir=cfg.get("result_dir", "./eval_results"),
        tta_runs=cfg.get("tta_runs", 5),
        topk=tuple(cfg.get("topk", [1, 5])),
        compute_calibration=cfg.get("compute_calibration", True),
        ood_detection=cfg.get("ood_detection", False),
        backbone=full_cfg.get("backbone", {}), # Use GLOBAL backbone config
        compute_mahalanobis=cfg.get("compute_mahalanobis", False),
        compute_vim=cfg.get("compute_vim", False),
    )
    
    # Needs num_classes
    num_classes = full_cfg.get("num_classes", 100)
    evaluator = Evaluator(loader, num_classes=num_classes, config=eval_cfg)
    
    # Load best model from arcface or distill snapshot
    # Priority: Distill Best > ArcFace Best > ArcFace Final
    # But usually we evaluate the "Main" model.
    # Where does ArcFace save? snapshots/backbone_best.pth or similar.
    # We should look in arcface snapshot dir.
    
    snapshot_dir = full_cfg.get("arcface", {}).get("snapshot_dir", "./snapshots")
    snapshot_path = os.path.join(snapshot_dir, "backbone_best.pth") # specific to ArcFaceTrainer
    
    # Try project name based
    if "project_name" in full_cfg:
         p_path = os.path.join(snapshot_dir, f"{full_cfg['project_name']}_best.pth")
         if os.path.exists(p_path):
             snapshot_path = p_path
             
    if not os.path.exists(snapshot_path):
        LOGGER.warning(f"Feature Extractor snapshot not found at {snapshot_path}, checking final...")
        snapshot_path = os.path.join(snapshot_dir, "backbone_final.pth")

    if not os.path.exists(snapshot_path):
         LOGGER.error(f"FATAL: No model snapshot found to evaluate at {snapshot_path}")
         return

    LOGGER.info(f"Loading model for evaluation from {snapshot_path}")
    
    # Helper to load header too? 
    # Evaluator needs head for loss/accuracy. 
    # ArcFaceTrainer saves head separately? usually 'head_best.pth'
    head_path = snapshot_path.replace("backbone", "head")
    if not os.path.exists(head_path):
        head_path = None
        
    model, head = evaluator.load_model(snapshot_path, head_path=head_path)
    evaluator.evaluate(model, head)


def evaluate_with_tta(cfg: dict, snapshot_dir: str):
    """
    Perform evaluation using Test Time Augmentation (TPA).
    By default, it uses the 'test' split from the garbage loader.
    """
    LOGGER.info("===> Starting TTA Evaluation")
    
    # Path logic
    project_name = cfg.get("project_name", "backbone")
    snapshot_path = os.path.join(snapshot_dir, f"{project_name}_best.pth")
    if not os.path.exists(snapshot_path):
        snapshot_path = os.path.join(snapshot_dir, "backbone_best.pth")
    
    if not os.path.exists(snapshot_path):
        LOGGER.error(f"No snapshot found for TTA evaluation at {snapshot_path}")
        return

    LOGGER.info(f"Loading snapshot from {snapshot_path}")

    # Configuration for Trainer (to get backbone/head)
    # Use global config for num_classes and backbone, as arcface section might not have all details
    trainer_cfg = ArcFaceConfig(
        num_classes=cfg.get("num_classes", 6),
        backbone=cfg.get("backbone", {}),
        image_size=cfg.get("image_size", 224),
    )
    
    # Initialize trainer without loaders just to get the model structure
    trainer = ArcFaceTrainer(train_loader=None, val_loader=None, config=trainer_cfg)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(snapshot_path, map_location=trainer.device)
        if "model_state_dict" in checkpoint: # EarlyStopping format
            trainer.backbone.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else: # Standard utils.save_snapshot format or direct backbone state_dict
            trainer.backbone.load_state_dict(checkpoint, strict=False)
        
        # Load head if separate
        head_path = snapshot_path.replace("backbone", "head")
        if os.path.exists(head_path):
            head_checkpoint = torch.load(head_path, map_location=trainer.device)
            if "model_state_dict" in head_checkpoint:
                trainer.head.load_state_dict(head_checkpoint["model_state_dict"], strict=False)
            else:
                trainer.head.load_state_dict(head_checkpoint, strict=False)
    except Exception as e:
        LOGGER.error(f"Failed to load weights for TTA: {e}")
        return

    trainer.backbone.eval()
    trainer.head.eval()
    
    device = trainer.device
    
    # Data Loader
    if "dataset" not in cfg:
        LOGGER.error("Dataset configuration missing for TTA.")
        return

    _, _, test_loader = create_garbage_loader(
        root_dirs=cfg["dataset"].get("root_dirs", []),
        batch_size=cfg["dataset"].get("batch_size", 32),
        num_workers=cfg["dataset"].get("num_workers", 4),
        val_split=0.0,
        test_split=cfg["dataset"].get("test_split", 0.1), # Use properly configured test split
        json_path=cfg["dataset"].get("json_path", None)
    )

    if not test_loader or len(test_loader) == 0:
        LOGGER.error("Test loader is empty. Check your dataset paths.")
        return

    all_preds = []
    all_labels = []
    
    LOGGER.info(f"Running TTA on {len(test_loader.dataset)} images...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # TTA: Original + Horizontal Flip
            feats1 = trainer.backbone(images)
            logits1 = trainer.head(feats1, labels=None) 
            
            images_flipped = torch.flip(images, dims=[3])
            feats2 = trainer.backbone(images_flipped)
            logits2 = trainer.head(feats2, labels=None)
            
            avg_logits = (logits1 + logits2) / 2
            preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    LOGGER.info(f"TTA Evaluation Accuracy: {acc*100:.2f}%")
    LOGGER.info(f"TTA Evaluation F1 Score (Macro): {f1:.4f}")


def run_pipeline(config_path: str, phases: List[str]) -> None:

    # Auto-generate unique log file
    import datetime
    os.makedirs("./logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./logs/pipeline_{timestamp}.log"
    print(f"Logging to: {os.path.abspath(log_file)}")
    
    setup_logger(log_file)
    
    LOGGER.info(f"Loading config from: {config_path}")
    cfg = load_config(config_path)

    phase_map = {
        "supcon": lambda: run_supcon_phase(cfg), # Pass FULL config to access dataset/backbone
        "arcface": lambda: run_arcface_phase(cfg), # Pass full cfg to arcface runner to access 'dataset'
        "distill": lambda: run_distill_phase(cfg), # Pass full cfg to access global backbone for teacher
        "evaluate": lambda: run_evaluation_phase(cfg), # Pass FULL config to access global props
        "tta": lambda: evaluate_with_tta(cfg, cfg.get("arcface", {}).get("snapshot_dir", "./snapshots")),
    }

    for phase in phases:
        runner = phase_map.get(phase.lower())
        if runner is None:
            LOGGER.warning("Unknown phase '%s', skipping.", phase)
            continue
        try:
            runner()
        except Exception as e:
            LOGGER.error(f"Phase {phase} failed: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image classification pipeline runner.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["supcon", "arcface", "distill", "evaluate", "tta"],
        default=["supcon", "arcface", "distill", "evaluate", "tta"],
        help="Pipeline phases to execute in order.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config, args.phases)
