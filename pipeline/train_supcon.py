from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from augmentations import build_train_transform
from backbone import BackboneConfig, HybridBackbone
from losses import SupConLoss
from pipeline.optimizers import Lookahead, ModelEMA, SAM, apply_gradient_centralization
from utils import init_wandb, save_snapshot

LOGGER = logging.getLogger(__name__)


@dataclass
class SupConConfig:
    temperature: float = 0.07
    num_views: int = 4
    lr: float = 1e-3
    rho: float = 0.05
    ema_decay: Optional[float] = 0.9995
    use_amp: bool = True
    steps: int = 100
    log_csv: str = "./logs/supcon_metrics.csv"
    snapshot_path: str = "./snapshots_supcon/final.pth"
    backbone: dict = field(default_factory=dict)
    image_size: int = 224
    augmentations: dict = field(default_factory=dict)
    num_workers: int = 4,
    wandb: dict = field(default_factory=dict)


class SupConPretrainer:
    def __init__(self, dataloader: DataLoader, config: Optional[SupConConfig] = None):
        self.cfg = config or SupConConfig()
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(os.path.dirname(self.cfg.log_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.snapshot_path), exist_ok=True)

        self.backbone_cfg = BackboneConfig(**self.cfg.backbone)
        self.model = HybridBackbone(self.backbone_cfg).to(self.device)
        self.loss_fn = SupConLoss(temperature=self.cfg.temperature)

        params = list(self.model.parameters())
        self.sam = SAM(
            params,
            torch.optim.AdamW,
            rho=self.cfg.rho,
            adaptive=True,
            lr=self.cfg.lr,
            weight_decay=1e-4,
        )
        apply_gradient_centralization(self.sam.base_optimizer)
        self.optimizer = Lookahead(self.sam)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)
        self.ema = ModelEMA(self.model, decay=self.cfg.ema_decay) if self.cfg.ema_decay else None
        self.wandb_run = init_wandb(self.cfg.wandb)
        self.view_transform = build_train_transform(
            self.cfg.augmentations,
            image_size=self.cfg.image_size,
            augment=True,
        )
        self.to_pil = transforms.ToPILImage()

        with open(self.cfg.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["step", "supcon_loss"])

    def multi_view_augment(self, x: torch.Tensor) -> torch.Tensor:
        pil = self.to_pil(x)
        return torch.stack([self.view_transform(pil) for _ in range(self.cfg.num_views)])

    def train(self):
        self.model.train()
        step_count = 0

        for images, labels in self.dataloader:
            if step_count >= self.cfg.steps:
                break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            view_list = [self.multi_view_augment(img.cpu()).to(self.device) for img in images]
            views = torch.cat(view_list, dim=0)
            expanded_labels = labels.repeat_interleave(self.cfg.num_views)

            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                feats = self.model(views)
                loss = self.loss_fn(feats, expanded_labels)

            if self.cfg.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.sam.base_optimizer)
            else:
                loss.backward()

            self.sam.first_step(zero_grad=True)

            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                feats = self.model(views)
                loss_second = self.loss_fn(feats, expanded_labels)

            if self.cfg.use_amp:
                self.scaler.scale(loss_second).backward()
                self.scaler.unscale_(self.sam.base_optimizer)
            else:
                loss_second.backward()

            self.sam.second_step(zero_grad=True, grad_scaler=self.scaler if self.cfg.use_amp else None)
            self.optimizer.update_slow()
            if self.cfg.use_amp:
                self.scaler.update()

            if self.ema:
                self.ema.update(self.model)

            step_count += 1
            LOGGER.info("Step %d: SupCon Loss = %.4f", step_count, loss_second.item())

            with open(self.cfg.log_csv, "a", newline="") as f:
                csv.writer(f).writerow([step_count, loss_second.item()])

            if self.wandb_run:
                self.wandb_run.log({"supcon/loss": loss_second.item()}, step=step_count)

        save_snapshot(self.model, epoch=step_count, folder=os.path.dirname(self.cfg.snapshot_path))
        torch.save(self.model.state_dict(), self.cfg.snapshot_path)
        if self.ema:
            ema_path = self.cfg.snapshot_path.replace(".pth", "_ema.pth")
            torch.save(self.ema.ema_model.state_dict(), ema_path)
        LOGGER.info("SupCon pretrained weights saved to %s", self.cfg.snapshot_path)
        if self.wandb_run:
            self.wandb_run.finish()


def create_supcon_loader(
    batch_size: int = 16,
    image_size: int = 224,
    augmentations: Optional[dict] = None,
    root: str = "./data",
    num_workers: int = 4,
) -> DataLoader:
    transform = build_train_transform(augmentations or {}, image_size=image_size, augment=True)
    dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    loader = create_supcon_loader()
    trainer = SupConPretrainer(loader, SupConConfig())
    trainer.train()
