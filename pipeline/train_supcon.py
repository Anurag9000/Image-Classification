from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets


from .backbone import BackboneConfig, HybridBackbone
from .losses import SupConLoss
from .optimizers import ModelEMA, SAM, apply_gradient_centralization
from utils.utils import save_snapshot

LOGGER = logging.getLogger(__name__)


@dataclass
class SupConConfig:
    temperature: float = 0.07
    num_views: int = 4
    lr: float = 1e-3
    steps: int = 200
    ema_decay: Optional[float] = 0.9995
    use_amp: bool = True
    snapshot_path: str = "./snapshots/supcon_final.pth"
    backbone: dict = field(default_factory=dict)
    image_size: int = 224
    augmentations: dict = field(default_factory=dict)
    num_workers: int = 4
    max_steps: Optional[int] = None
    use_sam: bool = False
    use_lookahead: bool = False
    rho: float = 0.05
    early_stopping_patience: int = 10
    val_split: float = 0.1


class SupConPretrainer:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader = None, config: Optional[SupConConfig] = None):
        from utils.utils import EarlyStopping
        
        self.cfg = config or SupConConfig()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(os.path.dirname(self.cfg.snapshot_path), exist_ok=True)

        self.backbone_cfg = BackboneConfig(**self.cfg.backbone)
        self.model = HybridBackbone(self.backbone_cfg).to(self.device)

        self.loss_fn = SupConLoss(temperature=self.cfg.temperature).to(self.device)

        # Optimizer Selection
        # Optimizer Selection with Differential Learning Rates
        # User Request: CBAM weighted higher (shifts more), Conv weighted lower (shifts less)
        
        main_params = []
        cbam_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "cbam" in name or "se_module" in name: # MobileNetV3 SE blocks are effectively attention
                cbam_params.append(param)
            else:
                main_params.append(param)
                
        # Group params
        grouped_params = [
            {"params": main_params, "lr": self.cfg.lr * 0.1}, # Backbone shifts slower
            {"params": cbam_params, "lr": self.cfg.lr * 10.0}, # CBAM/Attention shifts faster
        ]
        
        base_optimizer = torch.optim.AdamW(grouped_params, lr=self.cfg.lr, weight_decay=1e-4) # Base LR unused if groups set, but good fallback

        if hasattr(self.cfg, 'use_sam') and self.cfg.use_sam:
             self.sam = SAM(self.model.parameters(), base_optimizer=torch.optim.AdamW, lr=self.cfg.lr, weight_decay=1e-4, rho=self.cfg.rho)
             self.optimizer = self.sam
             LOGGER.info("Using SAM Optimizer for SupCon")
        else:
             self.sam = None
             self.optimizer = base_optimizer
             
        # Apply Gradient Centralization
        if self.sam:
            self.sam.base_optimizer = apply_gradient_centralization(self.sam.base_optimizer)
        else:
            self.optimizer = apply_gradient_centralization(self.optimizer)
        
        print("[DEBUG] Optimizer Initialized.", flush=True)

        if hasattr(self.cfg, 'use_lookahead') and self.cfg.use_lookahead:
            self.optimizer = Lookahead(self.optimizer, k=self.cfg.lookahead_k, alpha=self.cfg.lookahead_alpha)
            LOGGER.info(f"Using Lookahead Optimizer (k={self.cfg.lookahead_k}, alpha={self.cfg.lookahead_alpha})")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer if not self.sam else self.sam.base_optimizer,
            T_max=self.cfg.steps
        )
        # Fix: torch.cuda.amp.GradScaler -> torch.amp.GradScaler('cuda', ...)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.use_amp)
        self.model_ema = ModelEMA(self.model, decay=self.cfg.ema_decay) if self.cfg.ema_decay else None
        
        self.early_stopper = EarlyStopping(
            patience=self.cfg.early_stopping_patience, 
            verbose=True, 
            path=self.cfg.snapshot_path.replace(".pth", "_best.pth")
        )

        self.start_step = 0
        if hasattr(self.cfg, 'resume_from') and self.cfg.resume_from and os.path.exists(self.cfg.resume_from):
            LOGGER.info(f"RESUMING SUPCON from {self.cfg.resume_from}")
            ckpt = torch.load(self.cfg.resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                 self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'early_stop_state_dict' in ckpt:
                 self.early_stopper.load_state_dict(ckpt['early_stop_state_dict'])
            self.start_step = ckpt.get('step', 0)
            LOGGER.info(f"Successfully resumed SupCon at Step {self.start_step}")

        os.makedirs(os.path.dirname(self.cfg.snapshot_path), exist_ok=True)

    def _validate(self) -> float:
        if not self.val_loader:
             return 0.0
        
        self.model.eval()
        val_loss = 0.0
        steps = 0
        limit_batches = 20 # Optimization: Reduced to 20 for faster feedback
        
        with torch.no_grad():
             for i, (images, labels) in enumerate(self.val_loader):
                 if i >= limit_batches:
                     break
                 images = images.view(-1, 3, self.cfg.image_size, self.cfg.image_size).to(self.device)
                 expanded_labels = labels.view(-1, 1).repeat(1, self.cfg.num_views).view(-1).to(self.device)
                 
                 feats = self.model(images)
                 loss = self.loss_fn(feats, expanded_labels)
                 val_loss += loss.item()
                 steps += 1
        
        self.model.train()
        return val_loss / max(steps, 1)

    def train(self):
        self.model.train()
        step = self.start_step
        data_iter = iter(self.train_loader)
        LOGGER.info("[DEBUG] Data Iterator Created. Entering Loop...")

        while step < self.cfg.steps:
            if self.cfg.max_steps and step >= self.cfg.max_steps:
                break
            try:
                LOGGER.info(f"[TRACE] Step {step}: Requesting next batch...")
                images, labels = next(data_iter)
                LOGGER.info(f"[TRACE] Step {step}: Batch Loaded.")
            except StopIteration:
                LOGGER.info("[DEBUG] Iterator exhausted. Resetting...")
                data_iter = iter(self.train_loader)
                images, labels = next(data_iter)

            # Images is (B, V, C, H, W) where V is num_views
            batch_size = images.size(0)
            images = images.view(-1, 3, self.cfg.image_size, self.cfg.image_size).to(self.device)
            # Create expanded labels: each source label repeated num_views times
            expanded_labels = labels.view(-1, 1).repeat(1, self.cfg.num_views).view(-1).to(self.device)
            LOGGER.info(f"[TRACE] Step {step}: Data moved to GPU.")

            # Fix: torch.cuda.amp.autocast -> torch.amp.autocast('cuda', ...)
            with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                feats = self.model(images)
                if torch.isnan(feats).any():
                     print("NaN detected in SupCon Model Output (Features)!")
                     raise RuntimeError("NaN detected in SupCon Model Output (Features)!")
                
                loss = self.loss_fn(feats, expanded_labels)
                if torch.isnan(loss):
                     print("NaN detected in SupCon Loss!")
                     raise RuntimeError("NaN detected in SupCon Model Output (Features)!")

            if self.sam:
                 self.scaler.unscale_(self.sam.base_optimizer)
                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                 
                 def closure_first():
                     self.sam.first_step(zero_grad=True)
                     return loss
                 
                 closure_first()
                 
                 with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                     feats_2 = self.model(images)
                     loss_2 = self.loss_fn(feats_2, expanded_labels)
                     
                 self.scaler.scale(loss_2).backward()
                 self.scaler.unscale_(self.sam.base_optimizer)
                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                 
                 self.sam.second_step(zero_grad=True, grad_scaler=self.scaler)
                 self.scaler.update()

            elif self.cfg.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            loss_second = loss # Legacy compatibility for logging, as SAM had a second loss

            if self.model_ema:
                self.model_ema.update(self.model)

            # Memory Cleanup
            # Memory Cleanup
            # if step % 200 == 0:
            #      torch.cuda.empty_cache()
            #      import gc
            #      gc.collect()

            if step % 10 == 0:
                 LOGGER.info(f"[HEARTBEAT-SUPCON] Step {step}/{self.cfg.steps} (Loss: {loss.item():.4f})")
            
            # Optional: Automatic recovery from OOM could go here
            # try: ... except RuntimeError: ...

            step += 1
            if step % 100 == 0:
                # 1. Run Configured Validation (Subset)
                val_loss_str = "N/A"
                patience_str = "N/A"
                

                if self.val_loader:
                     val_loss = self._validate()
                     val_loss_str = f"{val_loss:.4f}"
                     
                     # Early Stopping check
                     if self.early_stopper:
                        checkpoint_data = {
                            "step": step,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "early_stop_state_dict": self.early_stopper.state_dict() if hasattr(self.early_stopper, 'state_dict') else None
                        }
                        self.early_stopper(val_loss, checkpoint_data)
                        patience_str = f"{self.early_stopper.counter}/{self.early_stopper.patience}"
                          
                        if self.early_stopper.early_stop:
                            LOGGER.info(f"SupCon Step [{step}/{self.cfg.steps}] - Loss: {loss_second.item():.4f} - ValLoss: {val_loss_str} - Patience: {patience_str} [STOP TRIGGERED]")
                            LOGGER.info("Early stopping triggered in SupCon Phase!")
                            break
                
                # Combined Log Line
                LOGGER.info(f"SupCon Step [{step}/{self.cfg.steps}] - Loss: {loss_second.item():.4f} - ValLoss: {val_loss_str} - Patience: {patience_str}")

        if not self.early_stopper or not self.early_stopper.early_stop:
             torch.save({"model_state_dict": self.model.state_dict(), "steps": self.cfg.steps}, self.cfg.snapshot_path)
             LOGGER.info("SupCon pretraining finished. Final model saved at %s", self.cfg.snapshot_path)



# Multi-View Dataset Logic
from .files_dataset import JsonDataset
from .augmentations import get_advanced_transforms

# Custom Transform Wrapper for MultiView
class MultiViewTransform:
    def __init__(self, transform, num_views=2):
        self.transform = transform
        self.num_views = num_views
    
    def __call__(self, image):
        # image is opencv numpy array
        views = []
        for _ in range(self.num_views):
            res = self.transform(image=image)
            views.append(res['image'])
        return torch.stack(views)

class AlbumentationsMultiViewAdapter:
    def __init__(self, transform, num_views=2):
        self.transform = transform
        self.num_views = num_views
    def __call__(self, image, **kwargs):
        views = []
        for _ in range(self.num_views):
            res = self.transform(image=image)
            views.append(res['image'])
        # Stack them: (V, C, H, W)
        return {'image': torch.stack(views)}

def create_supcon_loader(
    batch_size: int = 16,
    image_size: int = 224,
    augmentations: Optional[dict] = None,
    root: str = "./data",
    num_workers: int = 4,
    json_path: Optional[str] = None,
    num_views: int = 2
) -> DataLoader:
    # Use the shared advanced transforms but customized for Multi-View
    # Ideally SupCon needs specific TwoCropTransform.
    # For now, let's just use the robust training transform from files_dataset and apply it twice.
    import torch
    
    # Allow passing json_path via 'root' if it looks like a json, or explicit arg
    if root.endswith(".json"):
        json_path = root
        root_dir = os.path.dirname(root) # approximation
    else:
        root_dir = root

    raw_transform = get_advanced_transforms(is_training=True, img_size=image_size)
    
    # Use the global adapter class (pickleable)
    adapter = AlbumentationsMultiViewAdapter(raw_transform, num_views=num_views)
    
    if not json_path:
        raise ValueError("SupCon requires json_path for Dataset.")

    # Create dataset with MultiView Transform
    dataset = JsonDataset(json_path, root_dir=root_dir, transform=adapter)
    
    # Split into Train/Val
    val_size = int(len(dataset) * 0.1) # 10% valid
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        loader = create_supcon_loader()
        trainer = SupConPretrainer(loader, SupConConfig())
        # trainer.train() # Disabled by default to prevent accidental heavy runs
        LOGGER.info("SupCon Module Loaded Successfully. Run via run_pipeline.py for training.")
    except Exception as e:
        LOGGER.warning(f"Could not run SupCon smoke test: {e}")
