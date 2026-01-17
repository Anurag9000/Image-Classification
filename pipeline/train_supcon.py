from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from .augmentations import build_train_transform
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


class SupConPretrainer:
    def __init__(self, dataloader: DataLoader, config: Optional[SupConConfig] = None):
        self.cfg = config or SupConConfig()
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(os.path.dirname(self.cfg.snapshot_path), exist_ok=True)

        self.backbone_cfg = BackboneConfig(**self.cfg.backbone)
        self.model = HybridBackbone(self.backbone_cfg).to(self.device)

        self.loss_fn = SupConLoss(temperature=self.cfg.temperature).to(self.device)
        self.sam = SAM(
            self.model.parameters(),
            torch.optim.AdamW,
            rho=0.05,
            adaptive=True,
            lr=self.cfg.lr,
            weight_decay=1e-4,
        )
        apply_gradient_centralization(self.sam.base_optimizer)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.sam.base_optimizer, T_max=self.cfg.steps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)
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

            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                feats = self.model(images)
                loss = self.loss_fn(feats, expanded_labels)

            if self.cfg.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.sam.base_optimizer)
            else:
                loss.backward()

            self.sam.first_step(zero_grad=True)

            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                feats = self.model(images)
                loss_second = self.loss_fn(feats, expanded_labels)

            if self.cfg.use_amp:
                self.scaler.scale(loss_second).backward()
            else:
                loss_second.backward()

            self.sam.second_step(zero_grad=True, grad_scaler=self.scaler if self.cfg.use_amp else None)
            if self.cfg.use_amp:
                self.scaler.update()

            self.scheduler.step()
            if self.model_ema:
                self.model_ema.update(self.model)

            step += 1
            if step % 10 == 0:
                LOGGER.info("SupCon Step [%d/%d] - Loss: %.4f", step, self.cfg.steps, loss_second.item())

        torch.save(self.model.state_dict(), self.cfg.snapshot_path)
        LOGGER.info("SupCon pretraining finished. Final model saved at %s", self.cfg.snapshot_path)


def create_supcon_loader(
    batch_size: int = 16,
    image_size: int = 224,
    augmentations: Optional[dict] = None,
    root: str = "./data",
    num_workers: int = 4,
    json_path: Optional[str] = None
) -> DataLoader:
    # Use the shared garbage transforms but customized for Multi-View
    # Ideally SupCon needs specific TwoCropTransform.
    # For now, let's just use the robust training transform from files_dataset and apply it twice.
    from .files_dataset import get_garbage_transforms, JsonDataset, CombinedFilesDataset
    import torch
    
    transform = get_garbage_transforms(is_training=True, img_size=image_size)

    class MultiViewWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset, num_views=2):
            self.dataset = dataset
            self.num_views = num_views
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            # We need to access the underlying image before transform to apply random transform twice
            # But JsonDataset applies transform internally. This is a design flaw in JsonDataset for SupCon.
            # Workaround: Dataset returns transformed image. We cannot re-transform bits.
            # Fix: Create dataset with NO transform, then apply here.
            # But JsonDataset doesn't seemingly support 'no transform' easily without modifying it.
            # Let's Modify JsonDataset (in memory hack) or relies on files_dataset returning image if transform is None?
            # JsonDataset line 52: if self.transform: ... else: A.Compose...
            
            # Since we cannot easily access the raw image from the instantiated dataset without changing files_dataset,
            # We will rely on the fact that we fixed files_dataset to use whatever transform we pass.
            # But we passed `train_transform` to it.
            # Actually, we should instantiate JsonDataset with a transform that returns the raw image (or minimal resize), 
            # and then apply the heavy augs here.
            pass
            
            # Real Fix: Just instantiate Dataset with None, get raw image?
            # JsonDataset code: if transform is None, it applies Resize+ToTensor.
            # So we get a tensor. Tensors are hard to Augment with Albumentations (needs numpy).
            
            # Simpler Approach for now:
            # Just let the dataset assume it returns one view.
            # We will trick it? No.
            
            # FASTEST FIX: Instantiate JsonDataset with a special transform that generates 2 views?
            # No, transform interface is image->image.
            
            # Let's just trust files_dataset updates I made earlier? 
            # No, I didn't change JsonDataset to support MultiView.
            
            # Okay, I will use files_dataset directly but construct it here specially.
            raise NotImplementedError("SupCon logic requires raw image access. Use 'files_dataset.JsonDataset' directly.")

    # RE-IMPLEMENTING LOADER LOGIC FOR SUPCON CORRECTLY
    from .files_dataset import JsonDataset
    
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

    # Allow passing json_path via 'root' if it looks like a json, or explicit arg
    if root.endswith(".json"):
        json_path = root
        root_dir = os.path.dirname(root) # approximation
    else:
        root_dir = root

    raw_transform = get_garbage_transforms(is_training=True, img_size=image_size)
    mv_transform = MultiViewTransform(raw_transform, num_views=2)
    
    # We need a Dataset that DOES NOT apply transform itself, but lets us apply it.
    # JsonDataset from files_dataset applies transform.
    # We will subclass or specific usage.
    # Let's use JsonDataset but pass our MV transform!
    # JsonDataset calls transform(image=image). Our MV transform expects image=image.
    # BUT our MV returns a Tensor stack.
    # JsonDataset expects dict['image'] output from albumentations? 
    # Line 54: image = augmented['image']
    # So our transform must return {'image': TensorStack}.
    
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

    adapter = AlbumentationsMultiViewAdapter(raw_transform, num_views=2)
    
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

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    loader = create_supcon_loader()
    trainer = SupConPretrainer(loader, SupConConfig())
    trainer.train()
