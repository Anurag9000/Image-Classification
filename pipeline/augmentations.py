# augmentations.py

import random
from typing import Dict, Optional

import numpy as np
import torch
import torchvision.transforms.autoaugment as autoaugment
from torchvision import transforms
from torchvision.transforms import TrivialAugmentWide


# ------------------------------
# RandAugment, AugMix, AutoAugment pipelines
# ------------------------------
def get_randaugment_transform():
    return transforms.RandAugment()


def get_augmix_transform():
    return autoaugment.AugMix()


def get_autoaugment_transform():
    return autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET)


# ------------------------------
# MixUp
# ------------------------------
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ------------------------------
# CutMix
# ------------------------------
def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    cx = x.clone()
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    lam_adjusted = lam
    if cut_w > 0 and cut_h > 0:
        cx2 = x[index]  # Use original x for source patches

        cy = np.random.randint(0, h)
        cx_pos = np.random.randint(0, w)

        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        x1 = np.clip(cx_pos - cut_w // 2, 0, w)
        x2 = np.clip(cx_pos + cut_w // 2, 0, w)

        cx[:, :, y1:y2, x1:x2] = cx2[:, :, y1:y2, x1:x2]
        lam_adjusted = 1 - ((y2 - y1) * (x2 - x1) / (h * w))

    return cx, y, y[index], lam_adjusted


# ------------------------------
# TokenMix (Patch-level mixing)
# ------------------------------
def tokenmix(x, y, token_size=16):
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    cx = x.clone()
    rx = np.random.randint(0, w) // token_size * token_size
    ry = np.random.randint(0, h) // token_size * token_size
    cx[:, :, ry:ry + token_size, rx:rx + token_size] = cx[index, :, ry:ry + token_size, rx:rx + token_size]
    lam_adjusted = 1 - (token_size * token_size / (w * h))
    return cx, y, y[index], lam_adjusted


# ------------------------------
# MixToken (ViT token mixing)
# ------------------------------
def mixtoken(x, y, patch_size=16):
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    num_patches_h = max(1, h // patch_size)
    num_patches_w = max(1, w // patch_size)
    patch_idx_h = random.randint(0, num_patches_h - 1)
    patch_idx_w = random.randint(0, num_patches_w - 1)
    cx = x.clone()

    h_start = patch_idx_h * patch_size
    h_end = min((patch_idx_h + 1) * patch_size, h)
    w_start = patch_idx_w * patch_size
    w_end = min((patch_idx_w + 1) * patch_size, w)

    cx[:, :, h_start:h_end, w_start:w_end] = cx[index, :, h_start:h_end, w_start:w_end]
    lam_adjusted = 1 - ((h_end - h_start) * (w_end - w_start) / (h * w))
    return cx, y, y[index], lam_adjusted


# ------------------------------
# Unified Augmentation Switch
# ------------------------------
def mixup_cutmix_tokenmix(x, y, method='mixup'):
    if method == 'mixup':
        return mixup(x, y)
    elif method == 'cutmix':
        return cutmix(x, y)
    elif method == 'tokenmix':
        return tokenmix(x, y)
    elif method == 'mixtoken':
        return mixtoken(x, y)
    elif method in ('none', None):
        return x, y, y, 1.0
    else:
        raise ValueError(f"Unsupported mix method: {method}")


# ------------------------------
# Augmentations for TTA (Mixup + CutMix during inference)
# ------------------------------
def tta_mixup_cutmix(x, y):
    if random.random() < 0.5:
        return mixup(x, y)
    else:
        return cutmix(x, y)


# ------------------------------
# Transform builders
# ------------------------------
def build_train_transform(config: Optional[Dict] = None, image_size: int = 224, augment: bool = True):
    config = config or {}
    t = []
    mean = config.get("mean", [0.485, 0.456, 0.406])
    std = config.get("std", [0.229, 0.224, 0.225])

    if augment:
        if config.get("random_resized_crop", True):
            scale = config.get("rrc_scale", (0.7, 1.0))
            ratio = config.get("rrc_ratio", (0.75, 1.33))
            t.append(transforms.RandomResizedCrop(image_size, scale=scale, ratio=ratio))
        else:
            t.append(transforms.Resize((image_size, image_size)))

        if config.get("hflip", True):
            t.append(transforms.RandomHorizontalFlip())
        if config.get("vflip", False):
            t.append(transforms.RandomVerticalFlip())

        if config.get("color_jitter"):
            params = config["color_jitter"]
            t.append(
                transforms.ColorJitter(
                    brightness=params.get("brightness", 0.2),
                    contrast=params.get("contrast", 0.2),
                    saturation=params.get("saturation", 0.2),
                    hue=params.get("hue", 0.1),
                )
            )

        if config.get("autoaugment"):
            policy = config["autoaugment"]
            if isinstance(policy, str):
                policy = getattr(autoaugment.AutoAugmentPolicy, policy.upper(), autoaugment.AutoAugmentPolicy.IMAGENET)
            t.append(autoaugment.AutoAugment(policy))

        if config.get("randaugment"):
            params = config["randaugment"]
            t.append(
                transforms.RandAugment(
                    num_ops=params.get("num_ops", 2),
                    magnitude=params.get("magnitude", 9),
                )
            )

        if config.get("trivialaugment"):
            t.append(TrivialAugmentWide())
    else:
        t.append(transforms.Resize((image_size, image_size)))

    t.append(transforms.ToTensor())
    if config.get("normalize", True):
        t.append(transforms.Normalize(mean=mean, std=std))

    if augment and config.get("random_erasing"):
        params = config["random_erasing"]
        t.append(
            transforms.RandomErasing(
                p=params.get("p", 0.25),
                scale=params.get("scale", (0.02, 0.33)),
                ratio=params.get("ratio", (0.3, 3.3)),
            )
        )

    return transforms.Compose(t)


def build_eval_transform(config: Optional[Dict] = None, image_size: int = 224):
    config = config or {}
    mean = config.get("mean", [0.485, 0.456, 0.406])
    std = config.get("std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


if __name__ == "__main__":
    dummy_x = torch.randn(8, 3, 224, 224).cuda()
    dummy_y = torch.randint(0, 100, (8,)).cuda()

    mx, ya, yb, lam = mixup(dummy_x, dummy_y)
    print("MixUp:", mx.shape, lam)

    cx, ya, yb, lam = cutmix(dummy_x, dummy_y)
    print("CutMix:", cx.shape, lam)

    tx, ya, yb, lam = tokenmix(dummy_x, dummy_y)
    print("TokenMix:", tx.shape, lam)

    mt_x, ya, yb, lam = mixtoken(dummy_x, dummy_y)
    print("MixToken:", mt_x.shape, lam)
