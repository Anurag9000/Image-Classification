"""
ArcFace training pipeline with modern optimization techniques.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from augmentations import build_train_transform, mixup_cutmix_tokenmix
from backbone import BackboneConfig, HybridBackbone
from losses import AdaFace, CurricularFace, EvidentialLoss, FocalLoss
from pipeline.optimizers import (
    Lookahead,
    ModelEMA,
    SAM,
    apply_gradient_centralization,
)
from utils import init_wandb, save_snapshot

LOGGER = logging.getLogger(__name__)


@dataclass
class ArcFaceConfig:
    num_classes: int = 100
    lr: float = 1e-4
    gamma: float = 2.0
    smoothing: float = 0.1
    epochs: int = 30
    mix_method: str = "mixup"
    use_curricularface: bool = False
    use_evidential: bool = False
    snapshot_dir: str = "./snapshots"
    log_csv: str = "./logs/train_metrics.csv"
    rho: float = 0.05
    ema_decay: Optional[float] = 0.9995
    grad_clip_norm: Optional[float] = 1.0
    use_amp: bool = True
    compile_model: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    backbone: dict = field(default_factory=dict)
    image_size: int = 224
    augmentations: dict = field(default_factory=dict)
    use_manifold_mixup: bool = False
    manifold_mixup_alpha: float = 2.0
    manifold_mixup_weight: float = 0.5
    wandb: dict = field(default_factory=dict)


class ArcFaceTrainer:
    def __init__(self, dataloader: DataLoader, config: Optional[ArcFaceConfig] = None):
        self.cfg = config or ArcFaceConfig()
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(os.path.dirname(self.cfg.log_csv), exist_ok=True)
        os.makedirs(self.cfg.snapshot_dir, exist_ok=True)

        self.backbone_cfg = BackboneConfig(**self.cfg.backbone)
        self.backbone = HybridBackbone(self.backbone_cfg).to(self.device)
        if self.cfg.compile_model and hasattr(torch, "compile"):
            self.backbone = torch.compile(self.backbone)  # type: ignore[assignment]

        if self.cfg.use_curricularface:
            self.head = CurricularFace(embedding_size=self.backbone_cfg.fusion_dim, num_classes=self.cfg.num_classes).to(self.device)
        else:
            self.head = AdaFace(embedding_size=self.backbone_cfg.fusion_dim, num_classes=self.cfg.num_classes).to(self.device)

        params = list(self.backbone.parameters()) + list(self.head.parameters())
        self.trainable_params = params
        self.sam = SAM(
            self.trainable_params,
            torch.optim.AdamW,
            rho=self.cfg.rho,
            adaptive=True,
            lr=self.cfg.lr,
            weight_decay=1e-4,
        )
        apply_gradient_centralization(self.sam.base_optimizer)
        self.optimizer = Lookahead(self.sam, k=self.cfg.lookahead_k, alpha=self.cfg.lookahead_alpha)

        total_steps = self.cfg.epochs * len(self.dataloader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.sam.base_optimizer,
            max_lr=self.cfg.lr * 10,
            total_steps=total_steps,
        )

        if self.cfg.use_evidential:
            self.criterion = EvidentialLoss(num_classes=self.cfg.num_classes)
        else:
            self.criterion = FocalLoss(alpha=1.0, gamma=self.cfg.gamma, smoothing=self.cfg.smoothing)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)
        self.backbone_ema = ModelEMA(self.backbone, decay=self.cfg.ema_decay) if self.cfg.ema_decay else None
        self.head_ema = ModelEMA(self.head, decay=self.cfg.ema_decay) if self.cfg.ema_decay else None
        self.manifold_mixup_enabled = self.cfg.use_manifold_mixup
        self.wandb_run = init_wandb(self.cfg.wandb)

        with open(self.cfg.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "train_f1"])

    def _update_ema(self):
        if self.backbone_ema:
            self.backbone_ema.update(self.backbone)
        if self.head_ema:
            self.head_ema.update(self.head)

    def _compute_loss(self, logits, y_a, y_b, lam):
        return lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)

    def _apply_manifold_mixup(self, logits, labels):
        lam = np.random.beta(self.cfg.manifold_mixup_alpha, self.cfg.manifold_mixup_alpha)
        idx = torch.randperm(logits.size(0), device=logits.device)
        mixed_logits = lam * logits + (1 - lam) * logits[idx]
        return self._compute_loss(mixed_logits, labels, labels[idx], lam)

    def train(self):
        self.backbone.train()
        self.head.train()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_losses = []
            preds_all, labels_all = [], []

            for images, labels in self.dataloader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                mixed_x, y_a, y_b, lam = mixup_cutmix_tokenmix(images.clone(), labels, method=self.cfg.mix_method)

                with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                    features = self.backbone(mixed_x)
                    logits = self.head(features, y_a)
                    loss = self._compute_loss(logits, y_a, y_b, lam)
                    if self.manifold_mixup_enabled:
                        mix_loss = self._apply_manifold_mixup(logits, y_a)
                        loss = (loss + mix_loss * self.cfg.manifold_mixup_weight) / (1 + self.cfg.manifold_mixup_weight)

                if self.cfg.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.sam.base_optimizer)
                else:
                    loss.backward()

                if self.cfg.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.cfg.grad_clip_norm)

                self.sam.first_step(zero_grad=True)

                with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                    features = self.backbone(mixed_x)
                    logits = self.head(features, y_a)
                    loss_second = self._compute_loss(logits, y_a, y_b, lam)
                    if self.manifold_mixup_enabled:
                        mix_loss = self._apply_manifold_mixup(logits, y_a)
                        loss_second = (loss_second + mix_loss * self.cfg.manifold_mixup_weight) / (1 + self.cfg.manifold_mixup_weight)

                if self.cfg.use_amp:
                    self.scaler.scale(loss_second).backward()
                    self.scaler.unscale_(self.sam.base_optimizer)
                else:
                    loss_second.backward()

                if self.cfg.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.cfg.grad_clip_norm)

                self.sam.second_step(zero_grad=True, grad_scaler=self.scaler if self.cfg.use_amp else None)
                self.optimizer.update_slow()

                if self.cfg.use_amp:
                    self.scaler.update()

                self.scheduler.step()
                self._update_ema()

                epoch_losses.append(loss_second.item())
                preds_all.extend(torch.argmax(logits.detach(), dim=1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

            acc = accuracy_score(labels_all, preds_all)
            f1 = f1_score(labels_all, preds_all, average="macro")
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

            LOGGER.info("Epoch %d/%d - Loss: %.4f | Acc: %.4f | F1: %.4f", epoch, self.cfg.epochs, avg_loss, acc, f1)

            with open(self.cfg.log_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, avg_loss, acc, f1])

            if self.wandb_run:
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, "get_last_lr") else self.cfg.lr
                self.wandb_run.log(
                    {
                        "arcface/train_loss": avg_loss,
                        "arcface/train_acc": acc,
                        "arcface/train_f1": f1,
                        "arcface/lr": current_lr,
                    },
                    step=epoch,
                )

            save_snapshot(self.backbone, epoch=epoch, folder=self.cfg.snapshot_dir)
            save_snapshot(self.head, epoch=epoch, folder=os.path.join(self.cfg.snapshot_dir, "head"))
            if self.backbone_ema and self.head_ema:
                save_snapshot(self.backbone_ema.ema_model, epoch=epoch, folder=os.path.join(self.cfg.snapshot_dir, "ema", "backbone"))
                save_snapshot(self.head_ema.ema_model, epoch=epoch, folder=os.path.join(self.cfg.snapshot_dir, "ema", "head"))

        if self.wandb_run:
            self.wandb_run.finish()


def create_dataloader(
    batch_size: int = 32,
    augment: bool = True,
    image_size: int = 224,
    augmentations: Optional[dict] = None,
    root: str = "./data",
) -> DataLoader:
    transform = build_train_transform(augmentations or {}, image_size=image_size, augment=augment)
    dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    dataloader = create_dataloader(batch_size=32, augment=True)
    trainer = ArcFaceTrainer(dataloader, ArcFaceConfig(use_curricularface=True))
    trainer.train()
