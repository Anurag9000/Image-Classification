# augmentations.py

import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import random
import torchvision.transforms.autoaugment as autoaugment


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
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ------------------------------
# CutMix
# ------------------------------
def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    rx = np.random.randint(w)
    ry = np.random.randint(h)
    rw = np.random.randint(w // 2)
    rh = np.random.randint(h // 2)
    x[:, :, ry:ry+rh, rx:rx+rw] = x[index, :, ry:ry+rh, rx:rx+rw]
    lam_adjusted = 1 - (rw * rh / (w * h))
    return x, y, y[index], lam_adjusted


# ------------------------------
# TokenMix (Patch-level mixing)
# ------------------------------
def tokenmix(x, y, token_size=16):
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    rx = np.random.randint(0, w, size=(1,)) // token_size * token_size
    ry = np.random.randint(0, h, size=(1,)) // token_size * token_size
    x[:, :, ry:ry+token_size, rx:rx+token_size] = x[index, :, ry:ry+token_size, rx:rx+token_size]
    lam_adjusted = 1 - (token_size * token_size / (w * h))
    return x, y, y[index], lam_adjusted


# ------------------------------
# MixToken (ViT token mixing)
# ------------------------------
def mixtoken(x, y, patch_size=16):
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    patch_idx_h = random.randint(0, num_patches_h - 1)
    patch_idx_w = random.randint(0, num_patches_w - 1)

    x[:, :, patch_idx_h * patch_size: (patch_idx_h + 1) * patch_size,
      patch_idx_w * patch_size: (patch_idx_w + 1) * patch_size] = \
        x[index, :, patch_idx_h * patch_size: (patch_idx_h + 1) * patch_size,
          patch_idx_w * patch_size: (patch_idx_w + 1) * patch_size]

    lam_adjusted = 1 - (patch_size * patch_size / (h * w))
    return x, y, y[index], lam_adjusted


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
    else:
        raise ValueError("Unsupported mix method")


# ------------------------------
# Augmentations for TTA (Mixup + CutMix during inference)
# ------------------------------
def tta_mixup_cutmix(x, y):
    if random.random() < 0.5:
        return mixup(x, y)
    else:
        return cutmix(x, y)


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
