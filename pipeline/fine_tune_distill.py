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

from .augmentations import mixup_cutmix_tokenmix
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
    teacher_backbone_config: dict = field(default_factory=dict)
    val_split: float = 0.1
    use_sam: bool = False
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    val_limit_batches: int = 100

class FineTuneDistillTrainer:
    def __init__(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, config: Optional[DistillConfig] = None):
        self.cfg = config or DistillConfig()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.cfg.snapshot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.log_csv), exist_ok=True)

        self.backbone_cfg = BackboneConfig(**self.cfg.backbone) # Student Config
        self.backbone = HybridBackbone(self.backbone_cfg).to(self.device)
        self.head = AdaFace(embedding_size=self.backbone_cfg.fusion_dim, num_classes=self.cfg.num_classes).to(self.device)

        self.teacher_backbone: Optional[HybridBackbone] = None
        self.teacher_head: Optional[AdaFace] = None
        
        # Teacher Config Logic
        teacher_bb_cfg_dict = self.cfg.teacher_backbone_config if self.cfg.teacher_backbone_config else self.cfg.backbone
        self.teacher_backbone_cfg = BackboneConfig(**teacher_bb_cfg_dict)

        if self.cfg.teacher_backbone_path and os.path.exists(self.cfg.teacher_backbone_path):
            LOGGER.info("Loading teacher backbone from %s", self.cfg.teacher_backbone_path)
            state = torch.load(self.cfg.teacher_backbone_path, map_location=self.device)
            if "model_state_dict" in state:
                state = state["model_state_dict"] # Handle different save formats
                
            self.teacher_backbone = HybridBackbone(self.teacher_backbone_cfg).to(self.device)
            self.teacher_backbone.load_state_dict(state, strict=False)
            self.teacher_backbone.eval()
            for param in self.teacher_backbone.parameters():
                param.requires_grad = False
        else:
            LOGGER.warning("No teacher backbone path found or file does not exist. Distillation might fail if weights expected.")
        # Try to load head from backbone path if specific head path is missing
        if not self.cfg.teacher_head_path and self.cfg.teacher_backbone_path and os.path.exists(self.cfg.teacher_backbone_path):
             LOGGER.info("Attempting to load teacher head from backbone path: %s", self.cfg.teacher_backbone_path)
             try:
                 state = torch.load(self.cfg.teacher_backbone_path, map_location=self.device)
                 # Check if 'head' or 'head_state_dict' is in the checkpoint
                 head_state = None
                 if isinstance(state, dict):
                     if "head_state_dict" in state: head_state = state["head_state_dict"]
                     elif "head" in state: head_state = state["head"]
                 
                 if head_state:
                     self.teacher_head = AdaFace(embedding_size=self.teacher_backbone_cfg.fusion_dim, num_classes=self.cfg.num_classes).to(self.device)
                     self.teacher_head.load_state_dict(head_state, strict=False)
                     self.teacher_head.eval()
                     for param in self.teacher_head.parameters():
                         param.requires_grad = False
                     LOGGER.info("Successfully loaded teacher head from backbone checkpoint.")
             except Exception as e:
                 LOGGER.warning(f"Could not auto-load teacher head from backbone path: {e}")

        if self.cfg.teacher_head_path and os.path.exists(self.cfg.teacher_head_path):
            LOGGER.info("Loading teacher head from %s", self.cfg.teacher_head_path)
            state = torch.load(self.cfg.teacher_head_path, map_location=self.device)
            # Handle wrapped state dicts common in snapshots
            if isinstance(state, dict): 
                if "head_state_dict" in state: state = state["head_state_dict"]
                elif "head" in state: state = state["head"]
            
            self.teacher_head = AdaFace(embedding_size=self.teacher_backbone_cfg.fusion_dim, num_classes=self.cfg.num_classes).to(self.device)
            self.teacher_head.load_state_dict(state, strict=False)
            self.teacher_head.eval()
            for param in self.teacher_head.parameters():
                param.requires_grad = False

        # Save Teacher Model at the start for reference
        if self.teacher_backbone:
            save_snapshot(self.teacher_backbone, epoch=0, folder=os.path.join(self.cfg.snapshot_dir, "teacher_backbone"))
        if self.teacher_head:
            save_snapshot(self.teacher_head, epoch=0, folder=os.path.join(self.cfg.snapshot_dir, "teacher_head"))

        self.feature_proj = nn.Identity()
        if self.backbone_cfg.fusion_dim != self.teacher_backbone_cfg.fusion_dim:
            self.feature_proj = nn.Linear(self.backbone_cfg.fusion_dim, self.teacher_backbone_cfg.fusion_dim).to(self.device)
            LOGGER.info(f"Added projection layer: {self.backbone_cfg.fusion_dim} -> {self.teacher_backbone_cfg.fusion_dim}")

        params = [p for p in list(self.backbone.parameters()) + list(self.head.parameters()) + list(self.feature_proj.parameters()) if p.requires_grad]
        self.trainable_params = params
        
        # Optimizer Selection
        base_optimizer = torch.optim.AdamW(self.trainable_params, lr=self.cfg.lr, weight_decay=1e-4)
        
        if hasattr(self.cfg, 'use_sam') and self.cfg.use_sam:
             self.sam = SAM(self.trainable_params, base_optimizer=torch.optim.AdamW, lr=self.cfg.lr, weight_decay=1e-4, rho=self.cfg.rho)
             self.optimizer = self.sam 
             LOGGER.info("Using SAM Optimizer for Distillation")
        else:
             self.sam = None
             self.optimizer = base_optimizer
             
        self.optimizer = apply_gradient_centralization(self.optimizer)

        if hasattr(self.cfg, 'use_lookahead') and self.cfg.use_lookahead:
            self.optimizer = Lookahead(self.optimizer, k=self.cfg.lookahead_k, alpha=self.cfg.lookahead_alpha)
            LOGGER.info(f"Using Lookahead Optimizer (k={self.cfg.lookahead_k}, alpha={self.cfg.lookahead_alpha})")

        if self.cfg.max_steps and self.cfg.max_steps > 0:
            total_steps = self.cfg.epochs * min(len(self.train_loader), self.cfg.max_steps)
        else:
            total_steps = self.cfg.epochs * len(self.train_loader)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer if not self.sam else self.sam.base_optimizer,
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
        # Fix: torch.cuda.amp.GradScaler -> torch.amp.GradScaler('cuda', ...)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.use_amp)
        self.backbone_ema = ModelEMA(self.backbone, decay=self.cfg.ema_decay) if self.cfg.ema_decay else None
        self.head_ema = ModelEMA(self.head, decay=self.cfg.ema_decay) if self.cfg.ema_decay else None
        self.manifold_mixup_enabled = self.cfg.use_manifold_mixup
        self.wandb_run = init_wandb(self.cfg.wandb)
        
        self.start_epoch = 1
        self.step_resumed = 0
        if hasattr(self.cfg, "resume_from") and self.cfg.resume_from and os.path.exists(self.cfg.resume_from):
            LOGGER.info(f"RESUMING DISTILLATION from {self.cfg.resume_from}")
            ckpt = torch.load(self.cfg.resume_from, map_location=self.device)
            self.backbone.load_state_dict(ckpt['backbone_state_dict'])
            self.head.load_state_dict(ckpt['head_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'early_stop_state_dict' in ckpt:
                self.early_stopper.load_state_dict(ckpt['early_stop_state_dict'])
            self.start_epoch = ckpt.get('epoch', 1)
            self.step_resumed = ckpt.get('step', 0)
            LOGGER.info(f"Successfully resumed at Epoch {self.start_epoch}")

        with open(self.cfg.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "train_f1"])

    def _teacher_forward(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            teacher_backbone = self.teacher_backbone or (
                self.backbone_ema.ema_model if self.cfg.use_ema_teacher and self.backbone_ema else self.backbone
            )
            
            # CRITICAL: Do NOT fallback to student head if teacher backbone is present but head is missing.
            # This causes dimension mismatches (e.g. Teacher 1024 != Student 512).
            # Fallback only if we are in pure self-distillation mode (no external teacher backbone).
            if self.teacher_backbone:
                if self.teacher_head:
                    teacher_head = self.teacher_head
                else:
                    # If we have a teacher backbone but NO head, we cannot compute soft targets accurately.
                    # We should disable the head part of the forward pass or raise a warning/error.
                    # For now, let's raise RuntimeError to be safe as configured.
                    raise RuntimeError("Teacher backbone loaded but NO teacher head available! Provide 'teacher_head_path' or ensure checkpoint contains head weights.")
            else:
                teacher_head = self.teacher_head or (
                    self.head_ema.ema_model if self.cfg.use_ema_teacher and self.head_ema else self.head
                )

            backbone_mode = teacher_backbone.training
            head_mode = teacher_head.training
            teacher_backbone.eval()
            teacher_head.eval()

            features = teacher_backbone(images)
            # Use raw logits (labels=None) for distillation targets. 
            # We want the teacher's true distribution, not the margin-penalized training distribution.
            logits = teacher_head(features, labels=None)
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
        
        projected_student_features = self.feature_proj(student_features)
        feature_loss = self.feature_mse(projected_student_features, teacher_features)
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

    def _validate(self) -> Tuple[float, float, float, float]:
        if not self.val_loader:
            return 0.0, 0.0, 0.0, 0.0

        self.backbone.eval()
        self.head.eval()
        if self.teacher_backbone: self.teacher_backbone.eval()
        if self.teacher_head: self.teacher_head.eval()

        val_loss = 0.0
        val_steps = 0
        
        student_preds = []
        teacher_preds = []
        all_labels = []

        with torch.no_grad():
            limit_batches = getattr(self.cfg, "val_limit_batches", 100) # Increased default and made configurable
            for i, (images, labels) in enumerate(self.val_loader):
                if limit_batches and i >= limit_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Student Forward
                features = self.backbone(images)
                
                # 1. Loss needs Margins
                logits_loss = self.head(features, labels)
                loss = self.criterion(logits_loss, labels)
                val_loss += loss.item()
                val_steps += 1
                
                # 2. Metrics need Raw Cosine
                logits_raw = self.head(features, labels=None)
                student_preds.extend(torch.argmax(logits_raw, dim=1).cpu().numpy())
                
                # Teacher Forward (if exists)
                if self.teacher_backbone and self.teacher_head:
                    t_feats = self.teacher_backbone(images)
                    t_logits = self.teacher_head(t_feats, labels=None)
                    teacher_preds.extend(torch.argmax(t_logits, dim=1).cpu().numpy())
                
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / max(val_steps, 1)
        student_acc = accuracy_score(all_labels, student_preds)
        
        teacher_acc = 0.0
        if teacher_preds:
            teacher_acc = accuracy_score(all_labels, teacher_preds)

        self.backbone.train()
        self.head.train()
        return avg_val_loss, student_acc, teacher_acc

    def train(self):
        self.backbone.train()
        self.head.train()

        step_count = self.step_resumed
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            epoch_losses = []
            preds_all, labels_all = [], []

            for images, labels in self.train_loader:
                step_count += 1
                if self.cfg.max_steps and step_count > self.cfg.max_steps:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                teacher_features, teacher_soft = self._teacher_forward(images, labels)
                mixed_x, y_a, y_b, lam = mixup_cutmix_tokenmix(images.clone(), labels, method=self.cfg.mix_method)

                # Fix: torch.cuda.amp.autocast -> torch.amp.autocast('cuda', ...)
                with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                    student_features = self.backbone(mixed_x)
                    student_logits = self.head(student_features, y_a)
                    loss = self._distill_loss(student_logits, student_features, teacher_soft, teacher_features, y_a, y_b, lam)
                    if self.manifold_mixup_enabled:
                        mix_loss = self._apply_manifold_mixup(student_logits, y_a)
                        loss = (loss + mix_loss * self.cfg.manifold_mixup_weight) / (1 + self.cfg.manifold_mixup_weight)

                if self.sam:
                    self.scaler.unscale_(self.sam.base_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=5.0)
                    
                    def closure_first():
                        self.sam.first_step(zero_grad=True)
                        return loss 
                    
                    closure_first()
                    
                    with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                         # Re-compute for SAM step 2
                         student_features_2 = self.backbone(mixed_x)
                         student_logits_2 = self.head(student_features_2, y_a)
                         loss_2 = self._distill_loss(student_logits_2, student_features_2, teacher_soft, teacher_features, y_a, y_b, lam)
                         if self.manifold_mixup_enabled:
                             mix_loss_2 = self._apply_manifold_mixup(student_logits_2, y_a)
                             loss_2 = (loss_2 + mix_loss_2 * self.cfg.manifold_mixup_weight) / (1 + self.cfg.manifold_mixup_weight)

                    self.scaler.scale(loss_2).backward()
                    self.scaler.unscale_(self.sam.base_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=5.0)
                    
                    self.sam.second_step(zero_grad=True, grad_scaler=self.scaler)
                    self.scaler.update()

                elif self.cfg.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=5.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                loss_second = loss # Legacy compatibility

                # self.scheduler.step() moved to correct place?
                # OneCycleLR should step after optimizer
                self.scheduler.step()
                self._update_ema()

                epoch_losses.append(loss_second.item())
                preds_all.extend(torch.argmax(student_logits.detach(), dim=1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
 
                if step_count % 100 == 0:
                     # torch.cuda.empty_cache()
                     
                     # Frequent Validation Check (Every 10 steps)
                     val_loss_str = "N/A"
                     patience_str = "N/A"
                     student_acc_str = "N/A"
                     
                     if self.val_loader:
                         v_loss, v_acc, t_acc = self._validate()
                         val_loss_str = f"{v_loss:.4f}"
                         student_acc_str = f"{v_acc*100:.2f}%"
                         
                         if self.early_stopper:
                             self.early_stopper(v_loss, self.backbone)
                             patience_str = f"{self.early_stopper.counter}/{self.early_stopper.patience}"
                             
                             if self.early_stopper.early_stop:
                                 LOGGER.info(f"Epoch {epoch} Step [{len(epoch_losses)}/{len(self.train_loader)}] - Loss: {loss_second.item():.4f} - ValLoss: {val_loss_str} - Patience: {patience_str} [STOP TRIGGERED]")
                                 LOGGER.info("Early stopping triggered in Distill Phase!")
                                 break
                     
                     LOGGER.info(f"Epoch {epoch} Step [{len(epoch_losses)}/{len(self.train_loader)}] - Loss: {loss_second.item():.4f} - ValLoss: {val_loss_str} - Patience: {patience_str}")

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

            val_loss, val_acc, teacher_val_acc = self._validate()
            LOGGER.info(f"Epoch {epoch} Val Loss: {val_loss:.4f} | Student Val Acc: {val_acc:.4f} | Teacher Val Acc: {teacher_val_acc:.4f}")
            
            if self.wandb_run:
                self.wandb_run.log({
                    "distill/val_loss": val_loss,
                    "distill/val_acc": val_acc,
                    "distill/teacher_val_acc": teacher_val_acc
                }, step=epoch)

            checkpoint_data = {
                'epoch': epoch + 1,
                'step': step_count,
                'backbone_state_dict': self.backbone.state_dict(),
                'head_state_dict': self.head.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'early_stop_state_dict': self.early_stopper.state_dict() if hasattr(self.early_stopper, 'state_dict') else None
            }
            self.early_stopper(val_loss, checkpoint_data)
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
    json_path: Optional[str] = None,
    augment_online: bool = True
) -> tuple[DataLoader, Optional[DataLoader]]:
    from .files_dataset import create_data_loader
    
    # Just like ArcFace, unify data loading
    # Check if root is a list or string 
    root_dirs = [root] if isinstance(root, str) else root
    
    train_loader, val_loader, _ = create_data_loader(
        root_dirs=root_dirs,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        test_split=0.0,
        json_path=json_path,
        augment_online=augment_online,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        # train_loader, val_loader = create_distill_loader()
        # trainer = FineTuneDistillTrainer(train_loader, val_loader, DistillConfig())
        # trainer.train()
        LOGGER.info("Distill Module Loaded Successfully. Run via run_pipeline.py for training.")
    except Exception as e:
         LOGGER.warning(f"Could not run Distill smoke test: {e}")

