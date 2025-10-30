from __future__ import annotations

import argparse
import logging
from typing import List

import yaml
from torch.utils.data import DataLoader
from torchvision import datasets

from augmentations import build_eval_transform
from evaluate import EvaluationConfig, Evaluator
from fine_tune_distill import DistillConfig, FineTuneDistillTrainer, create_distill_loader
from train_arcface import ArcFaceConfig, ArcFaceTrainer, create_dataloader as create_arcface_loader
from train_supcon import SupConConfig, SupConPretrainer, create_supcon_loader
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
) -> DataLoader:
    transform = build_eval_transform(augmentations or {}, image_size=image_size)
    dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def run_supcon_phase(cfg: dict) -> None:
    if not cfg.get("enabled", True):
        LOGGER.info("SupCon phase disabled, skipping.")
        return

    LOGGER.info("===> Starting SupCon Pretraining Phase")
    loader = create_supcon_loader(
        batch_size=cfg.get("batch_size", 16),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations"),
        root=cfg.get("data_root", "./data"),
        num_workers=cfg.get("num_workers", 4),
    )
    supcon_cfg = SupConConfig(
        temperature=cfg.get("temperature", 0.07),
        num_views=cfg.get("num_views", 4),
        lr=cfg.get("lr", 1e-3),
        steps=cfg.get("steps", 200),
        ema_decay=cfg.get("ema_decay", 0.9995),
        backbone=cfg.get("backbone", {}),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations", {}),
        num_workers=cfg.get("num_workers", 4),
    )
    SupConPretrainer(loader, supcon_cfg).train()


def run_arcface_phase(cfg: dict) -> None:
    LOGGER.info("===> Starting ArcFace Training Phase")
    loader = create_arcface_loader(
        batch_size=cfg.get("batch_size", 32),
        augment=cfg.get("augment", True),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations"),
        root=cfg.get("data_root", "./data"),
    )
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
    )
    ArcFaceTrainer(loader, arcface_cfg).train()


def run_distill_phase(cfg: dict) -> None:
    if not cfg.get("enabled", True):
        LOGGER.info("Distillation phase disabled, skipping.")
        return

    LOGGER.info("===> Starting Fine-Tune + Distillation Phase")
    loader = create_distill_loader(
        batch_size=cfg.get("batch_size", 32),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations"),
        root=cfg.get("data_root", "./data"),
        num_workers=cfg.get("num_workers", 4),
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
        num_workers=cfg.get("num_workers", 4),
    )
    FineTuneDistillTrainer(loader, distill_cfg).train()


def run_evaluation_phase(cfg: dict) -> None:
    LOGGER.info("===> Starting Evaluation Phase")
    loader = create_eval_loader(
        batch_size=cfg.get("batch_size", 32),
        image_size=cfg.get("image_size", 224),
        augmentations=cfg.get("augmentations"),
        root=cfg.get("data_root", "./data"),
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


def run_pipeline(config_path: str, phases: List[str]) -> None:
    cfg = load_config(config_path)
    setup_logger(cfg.get("logging", {}).get("file", "./logs/full_pipeline.log"))

    phase_map = {
        "supcon": lambda: run_supcon_phase(cfg.get("supcon", {})),
        "arcface": lambda: run_arcface_phase(cfg.get("arcface", {})),
        "distill": lambda: run_distill_phase(cfg.get("distill", {})),
        "evaluate": lambda: run_evaluation_phase(cfg.get("evaluation", {})),
    }

    for phase in phases:
        runner = phase_map.get(phase.lower())
        if runner is None:
            LOGGER.warning("Unknown phase '%s', skipping.", phase)
            continue
        runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image classification pipeline runner.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["supcon", "arcface", "distill", "evaluate"],
        default=["supcon", "arcface", "distill", "evaluate"],
        help="Pipeline phases to execute in order.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config, args.phases)
