"""
Utility helpers for logging, checkpointing, and ensembling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def setup_logger(log_file: str) -> None:
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def save_checkpoint(state: dict, filename: str = "checkpoint.pth") -> None:
    torch.save(state, filename)
    logging.info("Checkpoint saved at %s", filename)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str = "checkpoint.pth", device: Optional[torch.device] = None) -> None:
    if not os.path.isfile(filename):
        logging.warning("No checkpoint found at %s", filename)
        return

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    logging.info("Checkpoint loaded from %s", filename)


def apply_swa(model: torch.nn.Module, swa_model: torch.nn.Module, swa_start: int, step: int, alpha: float = 0.01) -> None:
    if step < swa_start:
        return
    for swa_param, param in zip(swa_model.parameters(), model.parameters()):
        swa_param.data.mul_(1.0 - alpha).add_(param.data, alpha=alpha)


def save_snapshot(model: torch.nn.Module, epoch: int, folder: str = "./snapshots") -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    snapshot_file = os.path.join(folder, f"snapshot_epoch_{epoch}.pth")
    torch.save(model.state_dict(), snapshot_file)
    logging.info("Snapshot saved at %s", snapshot_file)


def apply_token_merging(vit_model: torch.nn.Module, ratio: float = 0.5) -> torch.nn.Module:
    try:
        import tome  # type: ignore
    except ImportError:
        logging.error("ToMe library not found. Install with: pip install git+https://github.com/GeorgeCazenavette/tome.git")
        return vit_model

    vit_model = tome.patch_vit(vit_model)
    vit_model.r = ratio
    logging.info("ToMe activated with ratio %.2f", ratio)
    return vit_model


def init_wandb(wandb_cfg: Optional[Dict]) -> Optional[Any]:
    if not wandb_cfg or not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        logging.warning("W&B not installed. Install with: pip install wandb")
        return None

    run = wandb.init(
        project=wandb_cfg.get("project", "image-classification"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        tags=wandb_cfg.get("tags"),
        config=wandb_cfg.get("config"),
        mode=wandb_cfg.get("mode", "online"),
        resume=wandb_cfg.get("resume", False),
    )
    logging.info("Initialised Weights & Biases run: %s", run.name)
    return run


if __name__ == "__main__":
    Path("./logs").mkdir(parents=True, exist_ok=True)
    setup_logger("./logs/training.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_model = torch.nn.Linear(10, 2).to(device)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())

    save_checkpoint(
        {
            "state_dict": dummy_model.state_dict(),
            "optimizer": dummy_optimizer.state_dict(),
        },
        filename="test_checkpoint.pth",
    )

    load_checkpoint(dummy_model, dummy_optimizer, filename="test_checkpoint.pth", device=device)
