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
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from .augmentations import mixup_cutmix_tokenmix
from .backbone import BackboneConfig, HybridBackbone
from .losses import AdaFace, CurricularFace, EvidentialLoss, FocalLoss
from .optimizers import (
    Lookahead,
    ModelEMA,
    SAM,
    apply_gradient_centralization,
)
from utils.utils import init_wandb, save_snapshot, EarlyStopping

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
    max_steps: Optional[int] = None
    wandb: dict = field(default_factory=dict)
    early_stopping_patience: int = 5
    val_split: float = 0.1
    use_sam: bool = False
    use_lookahead: bool = False
    supcon_snapshot: Optional[str] = None # Path to pretrained SupCon model to load
    resume_from: Optional[str] = None # Resume from checkpoint
    
    
    # ULMFiT Settings
    use_ulmfit: bool = False
    gradual_unfreezing: bool = False
    unfreeze_epoch: int = 2
    discriminative_lr_decay: float = 2.6
    val_limit_batches: Optional[int] = None
    rho: float = 0.05


class ArcFaceTrainer:
    def __init__(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, config: Optional[ArcFaceConfig] = None):
        self.cfg = config or ArcFaceConfig()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(os.path.dirname(self.cfg.log_csv), exist_ok=True)
        os.makedirs(self.cfg.snapshot_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.snapshot_dir, "head"), exist_ok=True)

        self.backbone_cfg = BackboneConfig(**self.cfg.backbone)
        self.backbone = HybridBackbone(self.backbone_cfg).to(self.device)
        if self.cfg.compile_model and hasattr(torch, "compile"):
            # WINDOWS WARNING: torch.compile is unstable on Windows and can hang indefinitely.
            # Disabling it by default unless user is on Linux.
            if os.name != 'nt':
                 self.backbone = torch.compile(self.backbone)
            else:
                 LOGGER.warning("Windows detected: Disabling torch.compile to prevent hangs.")

        # Load Pretrained SupCon Weights if available
        if self.cfg.supcon_snapshot and os.path.exists(self.cfg.supcon_snapshot):
            LOGGER.info(f"Loading SupCon weights from {self.cfg.supcon_snapshot}...")
            try:
                checkpoint = torch.load(self.cfg.supcon_snapshot, map_location=self.device)
                
                # Handle different saving formats (SupCon saves 'model_state_dict')
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                
                # Filter out incompatible keys (if any, though usually backbone matches)
                # SupCon model is HybridBackbone, ArcFace model is HybridBackbone. Checks out.
                # However, SupCon might be wrapped or have different prefixes if not careful.
                # Let's try strict=False to be safe but warn.
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                if missing:
                    LOGGER.warning(f"Missing keys when loading SupCon: {missing[:5]}...")
                if unexpected:
                    LOGGER.warning(f"Unexpected keys when loading SupCon: {unexpected[:5]}...")
                
                LOGGER.info("SupCon weights loaded successfully.")
            except Exception as e:
                LOGGER.error(f"Failed to load SupCon weights: {e}")

        if self.cfg.use_curricularface:
            self.head = CurricularFace(embedding_size=self.backbone_cfg.fusion_dim, num_classes=self.cfg.num_classes).to(self.device)
            LOGGER.info("Initialized CurricularFace Loss (Adaptive Curriculum).")
        else:
            self.head = AdaFace(embedding_size=self.backbone_cfg.fusion_dim, num_classes=self.cfg.num_classes).to(self.device)
            LOGGER.info("Initialized AdaFace Loss (Adaptive Margin for low-quality images).")

        params = [p for p in list(self.backbone.parameters()) + list(self.head.parameters()) if p.requires_grad]
        self.trainable_params = params
        
        # Optimizer Selection
        # Optimizer Selection with Differential Learning Rates
        main_params = []
        cbam_params = []
        head_params = []
        
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad: continue
            if "cbam" in name or "se_module" in name:
                cbam_params.append(param)
            else:
                main_params.append(param)
        
        for param in self.head.parameters():
            if param.requires_grad:
                head_params.append(param)
                
        if hasattr(self.cfg, 'use_ulmfit') and self.cfg.use_ulmfit:
            # Discriminative Learning Rates (ULMFiT Style)
            decay = getattr(self.cfg, 'discriminative_lr_decay', 2.6)
            LOGGER.info(f"ULMFiT Enabled: Applying Discriminative LRs with decay {decay}")
            
            # Simple assumption: Backbone -> Group params by "layer depth" roughly?
            # Or just separate Backbone vs Head.
            # Real ULMFiT slits the backbone into n groups. 
            # For EfficientNet/MobileNet, we can just treat the whole backbone as "lower learning rate" 
            # or try to split stages. 
            # Let's stick to the implementation from fine_tune_distill:
            # Group 1: Embeddings/Stem (Deepest) -> Low LR
            # Group 2: Middle Blocks -> Med LR
            # Group 3: Top Blocks -> Med-High LR
            # Group 4: Head -> Max LR
            
            # Simplified for Generic Backbone: 
            # Head = LR
            # Backbone = LR / Decay
            # CBAM = LR (High Freq needed)
            
            grouped_params = [
                {"params": main_params, "lr": self.cfg.lr / decay},
                {"params": cbam_params, "lr": self.cfg.lr}, # Keep CBAM fast
                {"params": head_params, "lr": self.cfg.lr},
            ]
        else:
            grouped_params = [
                {"params": main_params, "lr": self.cfg.lr * 0.1},
                {"params": cbam_params, "lr": self.cfg.lr * 10.0},
                {"params": head_params, "lr": self.cfg.lr},
            ]
        
        base_optimizer = torch.optim.AdamW(grouped_params, lr=self.cfg.lr, weight_decay=1e-4)
        
        if hasattr(self.cfg, 'use_sam') and self.cfg.use_sam: # Assuming use_sam flag, or add it to config
             # SAM requires a closure, handled in train loop.
             # But assuming we wrap the base optimizer.
             self.sam = SAM(grouped_params, base_optimizer_cls=torch.optim.AdamW, lr=self.cfg.lr, weight_decay=1e-4, rho=self.cfg.rho)
             self.optimizer = self.sam # Placeholder alias
             LOGGER.info("Using SAM Optimizer")
        else:
             self.sam = None
             self.optimizer = base_optimizer
             
        # Apply Gradient Centralization
        # If using SAM, we must apply GC to the BASE optimizer, because SAM calls base_optimizer.step()
        if self.sam:
            self.sam.base_optimizer = apply_gradient_centralization(self.sam.base_optimizer)
        else:
            self.optimizer = apply_gradient_centralization(self.optimizer)

        # Lookahead
        if hasattr(self.cfg, 'use_lookahead') and self.cfg.use_lookahead:
            self.optimizer = Lookahead(self.optimizer, k=self.cfg.lookahead_k, alpha=self.cfg.lookahead_alpha)
            LOGGER.info(f"Using Lookahead Optimizer (k={self.cfg.lookahead_k}, alpha={self.cfg.lookahead_alpha})")

        if self.train_loader:
            total_steps = self.cfg.epochs * len(self.train_loader)
        else:
            total_steps = 1
            
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer if not self.sam else self.sam.base_optimizer, 
            max_lr=self.cfg.lr * 10,
            total_steps=total_steps if total_steps > 0 else 1,
        )

        self.early_stopper = EarlyStopping(patience=self.cfg.early_stopping_patience, verbose=True, path=os.path.join(self.cfg.snapshot_dir, "best_model.pth"))

        if self.cfg.use_evidential:
            self.criterion = EvidentialLoss(num_classes=self.cfg.num_classes)
        else:
            self.criterion = FocalLoss(alpha=1.0, gamma=self.cfg.gamma, smoothing=self.cfg.smoothing)

        # Fix: torch.cuda.amp.GradScaler -> torch.amp.GradScaler('cuda', ...)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.use_amp)
        self.backbone_ema = ModelEMA(self.backbone, decay=self.cfg.ema_decay) if self.cfg.ema_decay else None
        self.head_ema = ModelEMA(self.head, decay=self.cfg.ema_decay) if self.cfg.ema_decay else None
        self.manifold_mixup_enabled = self.cfg.use_manifold_mixup
        
        self.start_epoch = 1
        self.step_resumed = 0
        
        # Resume Logic
        if self.cfg.resume_from and os.path.exists(self.cfg.resume_from):
            LOGGER.info(f"RESUMING TRAINING from {self.cfg.resume_from}")
            try:
                ckpt = torch.load(self.cfg.resume_from, map_location=self.device)
                
                # Load models
                self.backbone.load_state_dict(ckpt['backbone_state_dict'])
                self.head.load_state_dict(ckpt['head_state_dict'])
                
                # Load optimizer/states
                if 'optimizer_state_dict' in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    except Exception as opt_e:
                        LOGGER.warning(f"Could not load optimizer state (might be incompatible): {opt_e}. Starting with fresh optimizer.")
                
                if 'scheduler_state_dict' in ckpt:
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

                if 'early_stop_state_dict' in ckpt:
                    self.early_stopper.load_state_dict(ckpt['early_stop_state_dict'])
                
                self.start_epoch = ckpt.get('epoch', 1)
                self.step_resumed = ckpt.get('step', 0)
                
                LOGGER.info(f"Successfully resumed from Epoch {self.start_epoch}, Step {self.step_resumed}")
            except Exception as e:
                LOGGER.error(f"Failed to resume from checkpoint: {e}")
                # Don't raise, just warn and start fresh if it's just a state dict mismatch
                LOGGER.warning("Continuing with fresh optimizer/scheduler states.")

        # self.wandb_run = init_wandb(self.cfg.wandb)
        self.wandb_run = None # Force disable for debugging hangs

        mode = "a" if self.cfg.resume_from else "w"
        with open(self.cfg.log_csv, mode, newline="") as f:
            if mode == "w":
                csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_f1"])
            
        self.step_log_csv = os.path.join(os.path.dirname(self.cfg.log_csv), "train_steps.csv")
        with open(self.step_log_csv, mode, newline="") as f:
            if mode == "w":
                csv.writer(f).writerow(["epoch", "step", "loss", "acc", "lr"])

    def _update_ema(self):
        if self.backbone_ema:
            self.backbone_ema.update(self.backbone)
        if self.head_ema:
            self.head_ema.update(self.head)

    def _compute_loss(self, logits, y_a, y_b, lam):
        return lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)

    # _apply_cutmix and _rand_bbox removed as they are redundant with augmentations.py

    def _apply_manifold_mixup(self, logits, labels):
        lam = np.random.beta(self.cfg.manifold_mixup_alpha, self.cfg.manifold_mixup_alpha)
        idx = torch.randperm(logits.size(0), device=logits.device)
        mixed_logits = lam * logits + (1 - lam) * logits[idx]
        return self._compute_loss(mixed_logits, labels, labels[idx], lam)

    def _validate(self, loader=None):
        self.backbone.eval()
        self.head.eval()
        
        target_loader = loader if loader else self.val_loader
        val_loss = 0.0
        val_steps = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            # limit_batches = 50 # Removed hard limit for exhaustive validation accuracy
            limit_batches = getattr(self.cfg, "val_limit_batches", None) 
            for i, (images, labels) in enumerate(target_loader):
                if limit_batches and i >= limit_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                features = self.backbone(images)
                
                # 1. Loss needs Margins (Adaptive)
                logits_loss = self.head(features, labels) 
                loss = self.criterion(logits_loss, labels)
                val_loss += loss.item()
                val_steps += 1
                
                # 2. Metrics need Raw Cosine (No Margins)
                logits_raw = self.head(features, labels=None)
                preds = torch.argmax(logits_raw, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / max(val_steps, 1)
        avg_val_acc = accuracy_score(all_labels, all_preds) * 100.0 # Fix: scale to percentage
        avg_val_f1 = f1_score(all_labels, all_preds, average="macro")
        
        self.backbone.train()
        self.head.train()
        return avg_val_loss, avg_val_acc, avg_val_f1

    def test(self, test_loader):
        LOGGER.info("Starting Test Set Evaluation...")
        if not test_loader:
            LOGGER.warning("No Test Loader provided!")
            return

        # Load best model if available
        best_path = os.path.join(self.cfg.snapshot_dir, "best_model.pth")
        if os.path.exists(best_path):
             LOGGER.info(f"Loading best model from {best_path} for testing...")
             ckpt = torch.load(best_path, map_location=self.device)
             self.backbone.load_state_dict(ckpt['model_state_dict'])
             self.head.load_state_dict(ckpt['head_state_dict'])
        else:
             LOGGER.warning("Best model snapshot not found. Testing with current model state.")

        loss, acc, f1 = self._validate(loader=test_loader)
        LOGGER.info(f"\n{'='*20} TEST RESULTS {'='*20}")
        LOGGER.info(f"Test Loss:     {loss:.4f}")
        LOGGER.info(f"Test Accuracy: {acc:.2f}%")
        LOGGER.info(f"Test F1 Score: {f1:.4f}")
        LOGGER.info(f"{'='*54}\n")
        return acc, f1

    def train(self):
        self.backbone.train()
        self.head.train()
        
        step_count = self.step_resumed

        step_count = self.step_resumed

        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
             # --- ULMFiT: Gradual Unfreezing ---
            if hasattr(self.cfg, 'gradual_unfreezing') and self.cfg.gradual_unfreezing:
                unfreeze_epoch = getattr(self.cfg, 'unfreeze_epoch', 2) # Default to epoch 2 if not set
                
                if epoch < unfreeze_epoch:
                    LOGGER.info(f"ULMFiT: Epoch {epoch} - Freezing Backbone, Training Head Only")
                    self.backbone.eval()
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                    self.head.train()
                elif epoch == unfreeze_epoch:
                    LOGGER.info(f"ULMFiT: Epoch {epoch} - Unfreezing Backbone (Full Training)")
                    self.backbone.train()
                    for param in self.backbone.parameters():
                        param.requires_grad = True
                    # In true ULMFiT we unfreeze last layer then next etc. 
                    # For simplicity in this pivot: Epoch 1 Head, Epoch 2+ All.
            
            print(f"[HEARTBEAT] Starting Epoch {epoch}...")
            epoch_losses = []
            preds_all, labels_all = [], []
            # step_count not reset within epoch to track global steps correctly

            for images, labels in self.train_loader:
                # Heartbeat: Data Loaded
                print(f"[HEARTBEAT] Loaded Batch {step_count} (Time: {time.time()})", flush=True)
                step_count += 1
                if hasattr(self.cfg, "max_steps") and self.cfg.max_steps and step_count > self.cfg.max_steps:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if labels.max() >= self.cfg.num_classes or labels.min() < 0:
                    LOGGER.error(f"Labels out of range! Range: [{labels.min()}, {labels.max()}], num_classes: {self.cfg.num_classes}")
                    raise ValueError(f"Invalid labels detected: {labels}")

                if self.cfg.mix_method in ("cutmix", "cutmix_internal"):
                    from .augmentations import cutmix
                    mixed_x, y_a, y_b, lam = cutmix(images.clone(), labels, alpha=self.cfg.augmentations.get("cutmix_alpha", 1.0))
                else:
                    mixed_x, y_a, y_b, lam = mixup_cutmix_tokenmix(images.clone(), labels, method=self.cfg.mix_method)

                # DEBUG: Check inputs
                if torch.isnan(mixed_x).any() or torch.isinf(mixed_x).any():
                    LOGGER.error("NaN/Inf detected in INPUT IMAGES (mixed_x)")
                    raise RuntimeError("NaN/Inf detected in INPUT IMAGES (mixed_x)")

                # Fix: torch.cuda.amp.autocast -> torch.amp.autocast('cuda', ...)
                with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                    features = self.backbone(mixed_x)
                    if torch.isnan(features).any():
                        LOGGER.error("NaN detected in BACKBONE OUTPUT")
                        raise RuntimeError("NaN/Inf detected in INPUT IMAGES (mixed_x)")
                        
                    logits = self.head(features, y_a)
                    if torch.isnan(logits).any():
                        LOGGER.error("NaN detected in HEAD OUTPUT (Logits)")
                        raise RuntimeError("NaN/Inf detected in INPUT IMAGES (mixed_x)")

                    loss = self._compute_loss(logits, y_a, y_b, lam)
                    if torch.isnan(loss):
                        LOGGER.error(f"NaN detected in LOSS calculation. Logits max: {logits.max()}, min: {logits.min()}")
                        raise RuntimeError("NaN/Inf detected in INPUT IMAGES (mixed_x)")
                    
                    if self.manifold_mixup_enabled:
                        mix_loss = self._apply_manifold_mixup(logits, y_a)
                        loss = (loss + mix_loss * self.cfg.manifold_mixup_weight) / (1 + self.cfg.manifold_mixup_weight)

                # Calculate Batch Accuracy (Needed for every step CSV logging)
                with torch.no_grad():
                    clean_logits = self.head(features, labels=None) 
                    preds = torch.argmax(clean_logits, dim=1)
                    acc = (preds == labels).float().mean() * 100.0

                if step_count % 10 == 0:
                    # Frequent Logging but sparse validation
                    
                    # 1. Validation Check (Every 500 steps)
                    val_loss_str = "N/A"
                    patience_str = "N/A"
                    acc_str = f"{acc.item():.2f}%"
                    
                    if self.val_loader and step_count % 10 == 0:
                         print(f"[HEARTBEAT] Entering Validation at Step {step_count}...")
                         v_loss, v_acc, v_f1 = self._validate()
                         print(f"[HEARTBEAT] Exited Validation at Step {step_count}.")
                         val_loss_str = f"{v_loss:.4f}"
                         
                         # Early Stopping Check
                         if self.early_stopper:
                             self.early_stopper(v_loss, {
                                'model_state_dict': self.backbone.state_dict(),
                                'head_state_dict': self.head.state_dict()
                             })
                             patience_str = f"{self.early_stopper.counter}/{self.early_stopper.patience}"
                             
                             if self.early_stopper.early_stop:
                                 LOGGER.info(f"Epoch {epoch} Step [{step_count}/{len(self.train_loader)}] - Loss: {loss.item():.4f} - ValLoss: {val_loss_str} - Patience: {patience_str} [STOP TRIGGERED]")
                                 LOGGER.info("Early stopping triggered in ArcFace Phase!")
                                 break
                    
                    # Combined Log
                    LOGGER.info(f"Epoch {epoch} Step [{step_count}/{len(self.train_loader)}] - Loss: {loss.item():.4f} - Acc: {acc_str} - ValLoss: {val_loss_str} - Patience: {patience_str}")
                    
                    if acc.item() == 0.0:
                         LOGGER.warning("CRITICAL: Accuracy is EXACTLY 0.0!")

                # Backward & Optimizer Step
                if self.sam:
                    self.scaler.unscale_(self.sam.base_optimizer)
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.trainable_params, self.cfg.grad_clip_norm)
                    
                    def closure_first():
                        self.sam.first_step(zero_grad=True)
                        return loss
                    
                    closure_first()
                    
                    with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                         # Re-compute for SAM step 2
                         features_2 = self.backbone(mixed_x)
                         logits_2 = self.head(features_2, y_a)
                         loss_2 = self._compute_loss(logits_2, y_a, y_b, lam)
                         if self.manifold_mixup_enabled:
                             mix_loss_2 = self._apply_manifold_mixup(logits_2, y_a)
                             loss_2 = (loss_2 + mix_loss_2 * self.cfg.manifold_mixup_weight) / (1 + self.cfg.manifold_mixup_weight)

                    self.scaler.scale(loss_2).backward()
                    self.scaler.unscale_(self.sam.base_optimizer)
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.trainable_params, self.cfg.grad_clip_norm)
                    
                    self.sam.second_step(zero_grad=True, grad_scaler=self.scaler)
                    self.scaler.update()

                elif self.cfg.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.trainable_params, self.cfg.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.trainable_params, self.cfg.grad_clip_norm)
                    self.optimizer.step()

                # Zero grad for next step if not handled by SAM zero_grad interactions
                self.optimizer.zero_grad() 
                
                loss_second = loss 
                
                self.scheduler.step()
                self._update_ema()
                
                epoch_losses.append(loss.detach())

                # -------------------------------------------------------------------------
                # "Best Model Ever" Requirement: Step-wise Logging & Persistence
                # -------------------------------------------------------------------------
                # 1. Log every step to separate CSV
                with open(self.step_log_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([epoch, step_count, loss.item(), acc.item(), self.optimizer.param_groups[0]['lr']])

                # 2. Save "Latest" model every 100 steps
                # (Disabled to prevent Windows file locking/hangs)

            # End of Epoch
            avg_train_loss = np.mean([l.item() for l in epoch_losses]) if epoch_losses else 0.0
            val_loss, val_acc, val_f1 = self._validate()
            
            # Explicit Garbage Collection (Optional/Debug)
            # gc.collect()

            LOGGER.info(
                f"Epoch {epoch}/{self.cfg.epochs} - Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}"
            )

            # Save to CSV
            with open(self.cfg.log_csv, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, avg_train_loss, val_loss, val_acc, val_f1])

            # WandB logging
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "lr": self.optimizer.param_groups[0]['lr']
                })

            # Early Stopping and Checkpointing
            # Save both backbone and head for complete restoration/distillation
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
                LOGGER.info("Early stopping triggered.")
                break

            if epoch % 5 == 0:
                save_snapshot(self.backbone, epoch, folder=self.cfg.snapshot_dir)

        # Save Final models
        save_snapshot(self.backbone, self.cfg.epochs, folder=self.cfg.snapshot_dir)
        torch.save(self.backbone.state_dict(), os.path.join(self.cfg.snapshot_dir, "backbone_final.pth"))
        torch.save(self.head.state_dict(), os.path.join(self.cfg.snapshot_dir, "head", "head_final.pth"))
        LOGGER.info(f"Training finished. Final models saved in {self.cfg.snapshot_dir}")

def create_dataloader(
    batch_size: int = 32,
    augment: bool = True,
    image_size: int = 224,
    augmentations: Optional[dict] = None,
    root: str = "./data",
    num_workers: int = 4,
    val_split: float = 0.1,
) -> tuple[DataLoader, Optional[DataLoader]]:
    from .files_dataset import create_data_loader
    
    # Check if root is a list or string (create_data_loader expects list)
    root_dirs = [root] if isinstance(root, str) else root
    
    train_loader, val_loader, _ = create_data_loader(
        root_dirs=root_dirs,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        test_split=0.0, # No test split for training loader creation
    )

    return train_loader, val_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        # train_loader, val_loader = create_dataloader(batch_size=32, augment=True)
        # trainer = ArcFaceTrainer(train_loader, val_loader, ArcFaceConfig(use_curricularface=True))
        # trainer.train()
        LOGGER.info("ArcFace Module Loaded Successfully. Run via run_pipeline.py for training.")
    except Exception as e:
        LOGGER.warning(f"Could not run ArcFace smoke test: {e}")
