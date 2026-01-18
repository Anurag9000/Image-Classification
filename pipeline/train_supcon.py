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
from utils import save_snapshot

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
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5


class SupConPretrainer:
    def __init__(self, dataloader: DataLoader, config: Optional[SupConConfig] = None):
        self.cfg = config or SupConConfig()
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(os.path.dirname(self.cfg.snapshot_path), exist_ok=True)

        self.backbone_cfg = BackboneConfig(**self.cfg.backbone)
        self.model = HybridBackbone(self.backbone_cfg).to(self.device)

        self.loss_fn = SupConLoss(temperature=self.cfg.temperature).to(self.device)

        self.loss_fn = SupConLoss(temperature=self.cfg.temperature).to(self.device)

        # Optimizer Selection
        base_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-4)

        if hasattr(self.cfg, 'use_sam') and self.cfg.use_sam:
             self.sam = SAM(self.model.parameters(), base_optimizer=torch.optim.AdamW, lr=self.cfg.lr, weight_decay=1e-4, rho=self.cfg.rho)
             self.optimizer = self.sam
             LOGGER.info("Using SAM Optimizer for SupCon")
        else:
             self.sam = None
             self.optimizer = base_optimizer
             
        # Apply Gradient Centralization
        # Apply Gradient Centralization
        # Note: apply_gradient_centralization wraps the OPTIMIZER step, so we pass the optimizer.
        self.optimizer = apply_gradient_centralization(self.optimizer)

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

        os.makedirs(os.path.dirname(self.cfg.snapshot_path), exist_ok=True)

    def train(self):
        self.model.train()
        step = 0
        data_iter = iter(self.dataloader)

        while step < self.cfg.steps:
            if self.cfg.max_steps and step >= self.cfg.max_steps:
                break
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                images, labels = next(data_iter)

            # Images is (B, V, C, H, W) where V is num_views
            batch_size = images.size(0)
            images = images.view(-1, 3, self.cfg.image_size, self.cfg.image_size).to(self.device)
            # Create expanded labels: each source label repeated num_views times
            expanded_labels = labels.view(-1, 1).repeat(1, self.cfg.num_views).view(-1).to(self.device)

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
                 
                 self.sam.second_step(zero_grad=True)
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

            step += 1
            if step % 10 == 0:
                LOGGER.info("SupCon Step [%d/%d] - Loss: %.4f", step, self.cfg.steps, loss_second.item())

        torch.save({"model_state_dict": self.model.state_dict(), "steps": self.cfg.steps}, self.cfg.snapshot_path)
        LOGGER.info("SupCon pretraining finished. Final model saved at %s", self.cfg.snapshot_path)



# Multi-View Dataset Logic
from .files_dataset import JsonDataset
from .augmentations import get_garbage_transforms

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
    # Use the shared garbage transforms but customized for Multi-View
    # Ideally SupCon needs specific TwoCropTransform.
    # For now, let's just use the robust training transform from files_dataset and apply it twice.
    import torch
    
    # Allow passing json_path via 'root' if it looks like a json, or explicit arg
    if root.endswith(".json"):
        json_path = root
        root_dir = os.path.dirname(root) # approximation
    else:
        root_dir = root

    raw_transform = get_garbage_transforms(is_training=True, img_size=image_size)
    
    # Use the global adapter class (pickleable)
    adapter = AlbumentationsMultiViewAdapter(raw_transform, num_views=num_views)
    
    if json_path:
        # User explicitly passed json_path. 
        # root_dir should be the BASE directory where the file_paths in JSON are relative to.
        # Based on config, 'root' argument holds specific data dirs or "./data".
        # If 'root' is a list (from config format sometimes), take 1st, but here it is typed as str.
        # We trust the 'root' param passed in (which comes from cfg.get("data_root", "./data")).
        # However, run_pipeline passes cfg.get("data_root", "./data").
        # If JSON paths start with "Dataset_Final_Aug", and root is "./data", that is perfect.
        # DO NOT use os.path.dirname(json_path) as root_dir, that was the bug causing double path.
        dataset = JsonDataset(json_path, root_dir=root_dir, transform=adapter) 
    else:
        # Fallback to CIFAR if no JSON (unlikely for user)
        # But user HAS json.
        raise ValueError("SupCon requires json_path for Garbage dataset.")

    # IMPORTANT: Drop Last to avoid irregular batches messing up reshaping? 
    # Not strictly necessary but safe.
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    loader = create_supcon_loader()
    trainer = SupConPretrainer(loader, SupConConfig())
    trainer.train()
