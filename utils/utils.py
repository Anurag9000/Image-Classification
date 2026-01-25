"""
Utility helpers for logging, checkpointing, and ensembling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

LOGGER = logging.getLogger(__name__)


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

    if any(isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == os.path.abspath(log_file) for handler in logger.handlers):
        return

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Ensure no handlers are duplicated if called multiple times in same process
    unique_handlers = []
    seen_files = set()
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            abs_path = os.path.abspath(h.baseFilename)
            if abs_path not in seen_files:
                unique_handlers.append(h)
                seen_files.add(abs_path)
        else:
             # De-duplicate StreamHandlers too? Usually we need one.
             if not any(isinstance(uh, logging.StreamHandler) for uh in unique_handlers) or not isinstance(h, logging.StreamHandler):
                 unique_handlers.append(h)
    logger.handlers = unique_handlers


def save_checkpoint(state: dict, filename: str = "checkpoint.pth") -> None:
    torch.save(state, filename)
    logging.info("Checkpoint saved at %s", filename)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str = "checkpoint.pth", device: Optional[torch.device] = None) -> None:
    if not os.path.isfile(filename):
        logging.warning("No checkpoint found at %s", filename)
        return

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filename, map_location=device)
    
    # Robust loading: check for various possible keys
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("backbone")
        if not state_dict:
            # Maybe the checkpoint IS the state dict
            # Heuristic: check if keys look like layer names (strings)
            if all(isinstance(k, str) for k in checkpoint.keys()):
                state_dict = checkpoint
            else:
                state_dict = None
    else:
        state_dict = None

    if state_dict:
        model.load_state_dict(state_dict, strict=False)
    else:
        logging.warning("No valid state_dict found in %s", filename)

    if optimizer and "optimizer" in checkpoint:
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
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, snapshot_file)
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
    try:
        Path("./logs").mkdir(parents=True, exist_ok=True)
        # setup_logger("./logs/test_utils.log")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dummy Test
        LOGGER.info("Utils module loaded successfully.")
    except Exception as e:
        print(f"Utils smoke test failed: {e}")


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, path: str = "checkpoint.pth", verbose: bool = False, best_score: Optional[float] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = -best_score if best_score is not None else float("inf")

    def __call__(self, val_loss: float, model: torch.nn.Module):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model):
        if self.verbose:
            logging.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        
        if isinstance(model, dict):
            torch.save(model, self.path)
        else:
            torch.save({"model_state_dict": model.state_dict()}, self.path)
        
        self.val_loss_min = val_loss

    def state_dict(self):
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "val_loss_min": self.val_loss_min,
            "early_stop": self.early_stop
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        self.counter = state_dict.get("counter", 0)
        self.best_score = state_dict.get("best_score")
        self.val_loss_min = state_dict.get("val_loss_min", float("inf"))
        self.early_stop = state_dict.get("early_stop", False)

