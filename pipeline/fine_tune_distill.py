from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from .augmentations import build_train_transform, mixup_cutmix_tokenmix
from .backbone import BackboneConfig, HybridBackbone
from .losses import AdaFace, EvidentialLoss, FocalLoss
from .optimizers import Lookahead, ModelEMA, SAM, apply_gradient_centralization
from utils import init_wandb, save_snapshot, EarlyStopping

LOGGER = logging.getLogger(__name__)


@dataclass
class DistillConfig:
    num_classes: int = 100
    distill_weight: float = 0.3
    lr: float = 5e-6
    rho: float = 0.05
    mix_method: str = "mixup"
    use_evidential: bool = False
    temperature: float = 2.0
    ema_decay: Optional[float] = 0.9995
    use_amp: bool = True
    epochs: int = 10
    grad_clip_norm: Optional[float] = 1.0
    snapshot_dir: str = "./snapshots_distill"
    log_csv: str = "./logs/distill_metrics.csv"
    teacher_backbone_path: Optional[str] = None
    teacher_head_path: Optional[str] = None
    backbone: dict = field(default_factory=dict)
    image_size: int = 224
    augmentations: dict = field(default_factory=dict)
    use_manifold_mixup: bool = False
    manifold_mixup_alpha: float = 2.0
    manifold_mixup_weight: float = 0.5
    use_ema_teacher: bool = True
    num_workers: int = 4
    max_steps: Optional[int] = None
    wandb: dict = field(default_factory=dict)
    early_stopping_patience: int = 5
    val_split: float = 0.1


class FineTuneDistillTrainer:
    def __init__(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, config: Optional[DistillConfig] = None):
        self.cfg = config or DistillConfig()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.cfg.snapshot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.log_csv), exist_ok=True)

        self.backbone_cfg = BackboneConfig(**self.cfg.backbone)
        self.backbone = HybridBackbone(self.backbone_cfg).to(self.device)
        self.head = AdaFace(embedding_size=self.backbone_cfg.fusion_dim, num_classes=self.cfg.num_classes).to(self.device)

        self.teacher_backbone: Optional[HybridBackbone] = None
        self.teacher_head: Optional[AdaFace] = None
        if self.cfg.teacher_backbone_path and os.path.exists(self.cfg.teacher_backbone_path):
            LOGGER.info("Loading teacher backbone from %s", self.cfg.teacher_backbone_path)
            state = torch.load(self.cfg.teacher_backbone_path, map_location=self.device)
            self.backbone.load_state_dict(state, strict=False)
            self.teacher_backbone = HybridBackbone(self.backbone_cfg).to(self.device)
            self.teacher_backbone.load_state_dict(state, strict=False)
            self.teacher_backbone.eval()
            for param in self.teacher_backbone.parameters():
                param.requires_grad = False
        if self.cfg.teacher_head_path and os.path.exists(self.cfg.teacher_head_path):
            LOGGER.info("Loading teacher head from %s", self.cfg.teacher_head_path)
            state = torch.load(self.cfg.teacher_head_path, map_location=self.device)
            self.head.load_state_dict(state, strict=False)
            self.teacher_head = AdaFace(embedding_size=self.backbone_cfg.fusion_dim, num_classes=self.cfg.num_classes).to(self.device)
            self.teacher_head.load_state_dict(state, strict=False)
            self.teacher_head.eval()
            for param in self.teacher_head.parameters():
                param.requires_grad = False

        params = list(self.backbone.parameters()) + list(self.head.parameters())
        self.trainable_params = params
        self.sam = SAM(
            self.trainable_params,
            torch.optim.AdamW,
            rho=self.cfg.rho,
            adaptive=False,
            lr=self.cfg.lr,
            weight_decay=1e-4,
        )
        apply_gradient_centralization(self.sam.base_optimizer)
        self.optimizer = Lookahead(self.sam)

        total_steps = self.cfg.epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.sam.base_optimizer,
            max_lr=self.cfg.lr * 2,
            total_steps=total_steps if total_steps > 0 else 1,
        )

        self.early_stopper = EarlyStopping(patience=self.cfg.early_stopping_patience, verbose=True, path=os.path.join(self.cfg.snapshot_dir, "best_model.pth"))

        if self.cfg.use_evidential:
            self.criterion = EvidentialLoss(num_classes=self.cfg.num_classes)
        else:
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0, smoothing=0.1)

        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.feature_mse = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)
        self.backbone_ema = ModelEMA(self.backbone, decay=self.cfg.ema_decay) if self.cfg.ema_decay else None
        self.head_ema = ModelEMA(self.head, decay=self.cfg.ema_decay) if self.cfg.ema_decay else None
        self.manifold_mixup_enabled = self.cfg.use_manifold_mixup
        self.wandb_run = init_wandb(self.cfg.wandb)

        with open(self.cfg.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "train_f1"])

    def _teacher_forward(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            teacher_backbone = self.teacher_backbone or (
                self.backbone_ema.ema_model if self.cfg.use_ema_teacher and self.backbone_ema else self.backbone
            )
            teacher_head = self.teacher_head or (
                self.head_ema.ema_model if self.cfg.use_ema_teacher and self.head_ema else self.head
            )

            backbone_mode = teacher_backbone.training
            head_mode = teacher_head.training
            teacher_backbone.eval()
            teacher_head.eval()

            features = teacher_backbone(images)
            logits = teacher_head(features, labels)
            soft_targets = F.softmax(logits / self.cfg.temperature, dim=1)

            if backbone_mode:
                teacher_backbone.train()
            if head_mode:
                teacher_head.train()

        return features.detach(), soft_targets.detach()

    def _distill_loss(self, student_logits, student_features, teacher_soft, teacher_features, y_a, y_b, lam):
        hard_loss = lam * self.criterion(student_logits, y_a) + (1 - lam) * self.criterion(student_logits, y_b)
        soft_loss = self.kldiv(
            F.log_softmax(student_logits / self.cfg.temperature, dim=1),
            teacher_soft,
        ) * (self.cfg.temperature**2)
        feature_loss = self.feature_mse(student_features, teacher_features)
        return (
            (1 - self.cfg.distill_weight) * hard_loss
            + (self.cfg.distill_weight / 2) * soft_loss
            + (self.cfg.distill_weight / 2) * feature_loss
        )

    def _apply_manifold_mixup(self, logits, labels):
        lam = np.random.beta(self.cfg.manifold_mixup_alpha, self.cfg.manifold_mixup_alpha)
        idx = torch.randperm(logits.size(0), device=logits.device)
        mixed_logits = lam * logits + (1 - lam) * logits[idx]
        return lam * self.criterion(mixed_logits, labels) + (1 - lam) * self.criterion(mixed_logits, labels[idx])

    def _update_ema(self):
        if self.backbone_ema:
            self.backbone_ema.update(self.backbone)
        if self.head_ema:
            self.head_ema.update(self.head)

    def _validate(self) -> float:
        if not self.val_loader:
            return 0.0

        self.backbone.eval()
        self.head.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                features = self.backbone(images)
                logits = self.head(features, labels)
                # Validation metric: Just the main criterion (e.g. Focal/CrossEntropy) against labels
                loss = self.criterion(logits, labels)
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        self.backbone.train()
        self.head.train()
        return avg_val_loss

    def train(self):
        self.backbone.train()
        self.head.train()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_losses = []
            preds_all, labels_all = [], []
            step_count = 0

            for images, labels in self.train_loader:
                step_count += 1
                if self.cfg.max_steps and step_count > self.cfg.max_steps:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                teacher_features, teacher_soft = self._teacher_forward(images, labels)
                mixed_x, y_a, y_b, lam = mixup_cutmix_tokenmix(images.clone(), labels, method=self.cfg.mix_method)

                with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                    student_features = self.backbone(mixed_x)
                    student_logits = self.head(student_features, y_a)
                    loss = self._distill_loss(student_logits, student_features, teacher_soft, teacher_features, y_a, y_b, lam)
                    if self.manifold_mixup_enabled:
                        mix_loss = self._apply_manifold_mixup(student_logits, y_a)
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
                    student_features = self.backbone(mixed_x)
                    student_logits = self.head(student_features, y_a)
                    loss_second = self._distill_loss(
                        student_logits,
                        student_features,
                        teacher_soft,
                        teacher_features,
                        y_a,
                        y_b,
                        lam,
                    )
                    if self.manifold_mixup_enabled:
                        mix_loss = self._apply_manifold_mixup(student_logits, y_a)
                        loss_second = (loss_second + mix_loss * self.cfg.manifold_mixup_weight) / (1 + self.cfg.manifold_mixup_weight)

                if self.cfg.use_amp:
                    self.scaler.scale(loss_second).backward()
                else:
                    loss_second.backward()

                self.sam.second_step(zero_grad=True, grad_scaler=self.scaler if self.cfg.use_amp else None)
                self.optimizer.update_slow()
                if self.cfg.use_amp:
                    self.scaler.update()

                self.scheduler.step()
                self._update_ema()

                epoch_losses.append(loss_second.item())
                preds_all.extend(torch.argmax(student_logits.detach(), dim=1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
 
                if len(epoch_losses) % 10 == 0:
                    LOGGER.info(f"Epoch {epoch} Step [{len(epoch_losses)}/{len(self.dataloader)}] - Loss: {loss_second.item():.4f}")

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
                        "distill/train_loss": avg_loss,
                        "distill/train_acc": acc,
                        "distill/train_f1": f1,
                        "distill/lr": current_lr,
                    },
                    step=epoch,
                )

            save_snapshot(self.backbone, epoch=epoch, folder=os.path.join(self.cfg.snapshot_dir, "student_backbone"))
            save_snapshot(self.head, epoch=epoch, folder=os.path.join(self.cfg.snapshot_dir, "student_head"))
            if self.backbone_ema and self.head_ema:
                save_snapshot(
                    self.backbone_ema.ema_model,
                    epoch=epoch,
                    folder=os.path.join(self.cfg.snapshot_dir, "ema", "backbone"),
                )
                save_snapshot(
                    self.head_ema.ema_model,
                    epoch=epoch,
                    folder=os.path.join(self.cfg.snapshot_dir, "ema", "head"),
                )

            val_loss = self._validate()
            LOGGER.info(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")
            if self.wandb_run:
                self.wandb_run.log({"distill/val_loss": val_loss}, step=epoch)

            self.early_stopper(val_loss, self.backbone) # Saving backbone is enough, typically
            if self.early_stopper.early_stop:
                LOGGER.info("Early stopping triggered")
                break

        if self.wandb_run:
            self.wandb_run.finish()


def create_distill_loader(
    batch_size: int = 32,
    image_size: int = 224,
    augmentations: Optional[dict] = None,
    root: str = "./data",
    num_workers: int = 4,
    val_split: float = 0.1,
) -> tuple[DataLoader, Optional[DataLoader]]:
    transform = build_train_transform(augmentations or {}, image_size=image_size, augment=True)
    dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)

    if val_split > 0.0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_dataset else None

    return train_loader, val_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_loader, val_loader = create_distill_loader()
    trainer = FineTuneDistillTrainer(train_loader, val_loader, DistillConfig())
    trainer.train()

