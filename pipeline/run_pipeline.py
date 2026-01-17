from __future__ import annotations

import argparse
import logging
import os
import torch
from typing import List

import yaml
from torch.utils.data import DataLoader
from torchvision import datasets

from .augmentations import build_eval_transform
from .evaluate import EvaluationConfig, Evaluator
from .fine_tune_distill import DistillConfig, FineTuneDistillTrainer, create_distill_loader
from .train_arcface import ArcFaceConfig, ArcFaceTrainer, create_dataloader as create_arcface_loader
from .train_supcon import SupConConfig, SupConPretrainer, create_supcon_loader
from utils import setup_logger

LOGGER = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_eval_loader(
    batch_size: int = 32,
    image_size: int = 224,
    augmentations: dict | None = None,
    root: str = "./data",
    num_workers: int = 4,
) -> DataLoader:
    transform = build_eval_transform(augmentations or {}, image_size=image_size)
    dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


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

    loader = create_supcon_loader(
        batch_size=cfg.get("batch_size", 16),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations"),
        root=cfg.get("data_root", "./data"),
        num_workers=cfg.get("num_workers", 0),
        json_path=json_path,
        num_views=num_views 
    )
    supcon_cfg = SupConConfig(
        temperature=cfg.get("temperature", 0.07),
        num_views=num_views,
        lr=cfg.get("lr", 1e-3),
        steps=cfg.get("steps", 200),
        ema_decay=cfg.get("ema_decay", 0.9995),
        backbone=backbone_cfg,
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations", {}),
        num_workers=cfg.get("num_workers", 0),
        max_steps=cfg.get("max_steps"),
        snapshot_path=cfg.get("snapshot_path", "./snapshots/supcon_final.pth"),
    )
    SupConPretrainer(loader, supcon_cfg).train()


from pipeline.files_dataset import create_garbage_loader

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
            json_path=cfg["dataset"].get("json_path", None)
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

    arcface_cfg = ArcFaceConfig(
        num_classes=cfg.get("num_classes", 100),
        lr=cfg.get("lr", 1e-4),
        gamma=cfg.get("gamma", 2.0),
        smoothing=cfg.get("smoothing", 0.1),
        epochs=cfg.get("epochs", 30),
        mix_method=cfg.get("mix_method", "mixup"),
        use_curricularface=cfg.get("use_curricularface", True),
        use_evidential=cfg.get("use_evidential", False),
        ema_decay=cfg.get("ema_decay", 0.9995),
        compile_model=cfg.get("compile", False),
        backbone=cfg.get("backbone", {}),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations", {}),
        use_manifold_mixup=cfg.get("use_manifold_mixup", False),
        manifold_mixup_alpha=cfg.get("manifold_mixup_alpha", 2.0),
        manifold_mixup_weight=cfg.get("manifold_mixup_weight", 0.5),
        max_steps=cfg.get("max_steps"),
        snapshot_dir=cfg.get("snapshot_dir", "./snapshots"),
        log_csv=cfg.get("log_csv", "./logs/arcface_metrics.csv"),
        early_stopping_patience=cfg.get("early_stopping_patience", 5),
        val_split=cfg.get("val_split", 0.1),
        use_amp=cfg.get("use_amp", True),
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
        test_loss, test_acc = trainer._validate()
        
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
                 logits = trainer.head(feats, labels)
                 preds = torch.argmax(logits, dim=1).cpu().numpy()
                 all_preds.extend(preds)
                 all_labels.extend(labels.cpu().numpy())
                 
        from sklearn.metrics import accuracy_score, f1_score
        final_acc = accuracy_score(all_labels, all_preds)
        final_f1 = f1_score(all_labels, all_preds, average='macro')
        
        LOGGER.info(f"FINAL TEST RESULTS :: Accuracy: {final_acc:.4f} | F1 Score: {final_f1:.4f}")


def run_distill_phase(cfg: dict) -> None:
    if not cfg.get("enabled", True):
        LOGGER.info("Distillation phase disabled, skipping.")
        return

    LOGGER.info("===> Starting Fine-Tune + Distillation Phase")
    train_loader, val_loader = create_distill_loader(
        batch_size=cfg.get("batch_size", 32),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations"),
        root=cfg.get("data_root", "./data"),
        num_workers=cfg.get("num_workers", 0),
        val_split=cfg.get("val_split", 0.1),
    )
    distill_cfg = DistillConfig(
        num_classes=cfg.get("num_classes", 100),
        distill_weight=cfg.get("distill_weight", 0.3),
        lr=cfg.get("lr", 5e-6),
        mix_method=cfg.get("mix_method", "mixup"),
        epochs=cfg.get("epochs", 10),
        ema_decay=cfg.get("ema_decay", 0.9995),
        teacher_backbone_path=cfg.get("teacher_backbone_path"),
        teacher_head_path=cfg.get("teacher_head_path"),
        backbone=cfg.get("backbone", {}),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations", {}),
        use_manifold_mixup=cfg.get("use_manifold_mixup", False),
        manifold_mixup_alpha=cfg.get("manifold_mixup_alpha", 2.0),
        manifold_mixup_weight=cfg.get("manifold_mixup_weight", 0.5),
        use_ema_teacher=cfg.get("use_ema_teacher", True),
        num_workers=cfg.get("num_workers", 0),
        max_steps=cfg.get("max_steps"),
        snapshot_dir=cfg.get("snapshot_dir", "./snapshots_distill"),
        log_csv=cfg.get("log_csv", "./logs/distill_metrics.csv"),
        early_stopping_patience=cfg.get("early_stopping_patience", 5),
        val_split=cfg.get("val_split", 0.1),
    )
    FineTuneDistillTrainer(train_loader, val_loader, distill_cfg).train()


def run_evaluation_phase(cfg: dict) -> None:
    LOGGER.info("===> Starting Evaluation Phase")
    loader = create_eval_loader(
        batch_size=cfg.get("batch_size", 32),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations"),
        root=cfg.get("data_root", "./data"),
        num_workers=cfg.get("num_workers", 0),
    )
    eval_cfg = EvaluationConfig(
        result_dir=cfg.get("result_dir", "./eval_results"),
        tta_runs=cfg.get("tta_runs", 5),
        topk=tuple(cfg.get("topk", [1, 5])),
        compute_calibration=cfg.get("compute_calibration", True),
        ood_detection=cfg.get("ood_detection", False),
        backbone=cfg.get("backbone", {}),
    )

    evaluator = Evaluator(loader, num_classes=cfg.get("num_classes", 100), config=eval_cfg)
    backbone_path = cfg.get("backbone_path", "./snapshots/snapshot_epoch_30.pth")
    head_path = cfg.get("head_path", "./snapshots/head/snapshot_epoch_30.pth")
    model, head = evaluator.load_model(backbone_path, head_path=head_path)
    metrics = evaluator.evaluate(model, head)
    LOGGER.info("Evaluation metrics: %s", metrics)


def evaluate_with_tta(cfg: dict, snapshot_dir: str):
    from pipeline.files_dataset import create_garbage_test_loader
    from metrics.evaluator import Evaluator # Assuming Evaluator exists or we use sklearn directly as planned
    import torch
    import os
    import numpy as np
    from sklearn.metrics import accuracy_score
    from pipeline.train_arcface import ArcFaceTrainer
    from configs.config import ArcFaceConfig

    LOGGER.info("===> Starting TTA Evaluation")
    
    snapshot_path = os.path.join(snapshot_dir, f"{cfg.get('project_name', 'arcface')}_best.pth")
    if not os.path.exists(snapshot_path):
        LOGGER.warning(f"Best snapshot not found at {snapshot_path}, trying final...")
        snapshot_path = os.path.join(snapshot_dir, f"{cfg.get('project_name', 'arcface')}_final.pth")
    
    if not os.path.exists(snapshot_path):
        # Fallback to standard snapshot naming if project name based one missing
        snapshot_path = os.path.join(snapshot_dir, "backbone_best.pth")
        if not os.path.exists(snapshot_path):
             LOGGER.error("No snapshot found for TTA evaluation.")
             return

    LOGGER.info(f"Loading snapshot from {snapshot_path}")

    arc_cfg = cfg.get("arcface", {})
    trainer_cfg = ArcFaceConfig(
        num_classes=arc_cfg.get("num_classes", 6),
        backbone=arc_cfg.get("backbone", "resnet50"),
        embedding_size=arc_cfg.get("embedding_size", 512),
        image_size=arc_cfg.get("image_size", 224),
    )
    
    trainer = ArcFaceTrainer(dataloader=None, config=trainer_cfg)
    
    checkpoint = torch.load(snapshot_path, map_location=trainer.device)
    # Checkpoint loading logic similar to EarlyStopping or standard
    if "model_state_dict" in checkpoint: # EarlyStopping format
        trainer.backbone.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "head_state_dict" in checkpoint:
            trainer.head.load_state_dict(checkpoint["head_state_dict"], strict=False)
    elif "backbone" in checkpoint: # Standard utils.save_snapshot format
        trainer.backbone.load_state_dict(checkpoint["backbone"])
        if "head" in checkpoint:
            trainer.head.load_state_dict(checkpoint["head"])
    
    trainer.backbone.eval()
    trainer.head.eval()
    
    device = trainer.device
    
    # Check if dataset root_dirs exist
    if "dataset" not in cfg or "root_dirs" not in cfg["dataset"]:
        LOGGER.error("Dataset configuration missing for TTA.")
        return

    test_loader = create_garbage_test_loader(
        root_dirs=cfg["dataset"]["root_dirs"],
        batch_size=cfg["dataset"].get("batch_size", 32),
        num_workers=cfg["dataset"].get("num_workers", 4)
    )
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # TTA: Original + Flip
            feats1 = trainer.backbone(images)
            logits1 = trainer.head(feats1, labels) 
            
            images_flipped = torch.flip(images, dims=[3])
            feats2 = trainer.backbone(images_flipped)
            logits2 = trainer.head(feats2, labels)
            
            avg_logits = (logits1 + logits2) / 2
            preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    LOGGER.info(f"TTA Evaluation Accuracy: {acc:.4f}")


def run_pipeline(config_path: str, phases: List[str]) -> None:
    cfg = load_config(config_path)
    setup_logger(cfg.get("logging", {}).get("file", "./logs/full_pipeline.log"))

    phase_map = {
        "supcon": lambda: run_supcon_phase(cfg), # Pass FULL config to access dataset/backbone
        "arcface": lambda: run_arcface_phase(cfg), # Pass full cfg to arcface runner to access 'dataset'
        "distill": lambda: run_distill_phase(cfg.get("distill", {})),
        "evaluate": lambda: run_evaluation_phase(cfg.get("evaluation", {})),
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
