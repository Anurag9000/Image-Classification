# backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# ------------------------------
# Convolutional Block Attention Module (CBAM)
# ------------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Channel attention
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        channel_att = torch.sigmoid(self.fc(avg_pool) + self.fc(max_pool)).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x

# ------------------------------
# Dynamic Head Attention Fusion (DHA)
# ------------------------------
class DynamicHeadAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(DynamicHeadAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, vit_feat):
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        attn_weights = self.fc(combined)
        refined = combined * attn_weights
        return refined

# ------------------------------
# Hybrid Backbone: ConvNeXt-V2 + SwinV2 + CBAM + DHA Fusion
# ------------------------------
class HybridBackbone(nn.Module):
    def __init__(self, use_cbam=True):
        super(HybridBackbone, self).__init__()

        # ConvNeXt-V2 backbone (ImageNet-1K pretrained)
        self.cnn_backbone = create_model('convnextv2_base', pretrained=True)
        self.cnn_backbone.head = nn.Identity()

        # SwinV2 backbone (ImageNet-1K pretrained)
        self.vit_backbone = create_model('swinv2_base_window12_192_22k', pretrained=True)
        self.vit_backbone.head = nn.Identity()

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(1024)

        # Dynamic Head Attention for feature fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(1024 + 1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024)
        )
        self.dha = DynamicHeadAttention(1024 * 2)

    def forward(self, x):
        # CNN Features
        cnn_feat = self.cnn_backbone.forward_features(x)
        if self.use_cbam:
            cnn_feat_map = self.cnn_backbone.stages[-1](self.cnn_backbone.stages[-2](x))
            cnn_feat_map = self.cbam(cnn_feat_map)
            cnn_feat = F.adaptive_avg_pool2d(cnn_feat_map, 1).view(cnn_feat_map.size(0), -1)

        # ViT Features
        vit_feat = self.vit_backbone.forward_features(x)

        # Dynamic Head Attention Fusion
        refined_feat = self.dha(cnn_feat, vit_feat)
        fused = self.fusion_fc(refined_feat)
        return fused


if __name__ == "__main__":
    model = HybridBackbone(use_cbam=True).cuda()
    dummy = torch.randn(2, 3, 224, 224).cuda()
    out = model(dummy)
    print(f"Output Shape: {out.shape}")  # Should be (B, 1024)


# losses.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# AdaFace (Quality Adaptive Margin Loss)
# ------------------------------
class AdaFace(nn.Module):
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=64.0):
        super(AdaFace, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.s = scale
        self.m = margin
        self.eps = 1e-7

        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))
        self.momentum = 0.01

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        norm = torch.norm(embeddings, dim=1, keepdim=True).clamp(min=self.eps)

        with torch.no_grad():
            batch_mean = norm.mean()
            batch_std = norm.std()
            self.batch_mean = (1 - self.momentum) * self.batch_mean + self.momentum * batch_mean
            self.batch_std = (1 - self.momentum) * self.batch_std + self.momentum * batch_std

        margin_scaler = ((norm - self.batch_mean) / (self.batch_std + self.eps)).clamp(-1, 1)
        adaptive_m = self.m * (1 + margin_scaler)

        theta = torch.acos(cosine.clamp(-1 + self.eps, 1 - self.eps))
        target_logit = torch.cos(theta + adaptive_m.view(-1, 1))

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        output = cosine * (1 - one_hot) + target_logit * one_hot
        output *= self.s
        return output


# ------------------------------
# CurricularFace (Hard Sample Mining Adaptive ArcFace)
# ------------------------------
class CurricularFace(nn.Module):
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=64.0):
        super(CurricularFace, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.m = margin
        self.s = scale
        self.t = nn.Parameter(torch.ones(1) * 0.0)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_cos = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        hard_mask = cosine > target_cos
        self.t.data = 0.01 * cosine[hard_mask].mean().detach() + 0.99 * self.t.data if hard_mask.sum() > 0 else self.t.data

        output = torch.where(one_hot.bool(), target_cos, (cosine + self.t).clamp(-1, 1))
        output *= self.s
        return output


# ------------------------------
# Focal Loss with Smoothing
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.smoothing)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


# ------------------------------
# Supervised Contrastive Loss (SupCon)
# ------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits_mask = 1 - torch.eye(labels.shape[0], device=features.device)
        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        loss = - (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        return loss.mean()


# ------------------------------
# Evidential Deep Learning (EDL Loss for classification with uncertainty)
# ------------------------------
class EvidentialLoss(nn.Module):
    def __init__(self, num_classes, coeff=1.0):
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes
        self.coeff = coeff

    def forward(self, logits, labels):
        evidence = F.softplus(logits)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(logits.device)

        loglikelihood = torch.sum(one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        kl_div = self.coeff * self._kl_divergence(alpha, self.num_classes)
        return (kl_div - loglikelihood).mean()

    def _kl_divergence(self, alpha, num_classes):
        beta = torch.ones([1, num_classes], device=alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        kl = torch.lgamma(S_alpha) - torch.lgamma(S_beta) + torch.sum(torch.lgamma(beta) - torch.lgamma(alpha), dim=1, keepdim=True) + \
             torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha)), dim=1, keepdim=True)
        return kl.squeeze()


if __name__ == "__main__":
    dummy_feats = torch.randn(8, 1024).cuda()
    dummy_labels = torch.randint(0, 100, (8,)).cuda()

    adaface = AdaFace(1024, 100).cuda()
    logits = adaface(dummy_feats, dummy_labels)
    print("AdaFace logits:", logits.shape)

    currface = CurricularFace(1024, 100).cuda()
    logits_curr = currface(dummy_feats, dummy_labels)
    print("CurricularFace logits:", logits_curr.shape)

    supcon = SupConLoss().cuda()
    loss = supcon(dummy_feats, dummy_labels)
    print("SupCon Loss:", loss.item())

    focal = FocalLoss().cuda()
    ce_loss = focal(logits, dummy_labels)
    print("Focal Loss:", ce_loss.item())

    edl_loss_fn = EvidentialLoss(num_classes=100).cuda()
    edl_loss = edl_loss_fn(logits, dummy_labels)
    print("Evidential Loss:", edl_loss.item())

# train_arcface.py

import torch
import torch.nn as nn
import logging
from backbone import HybridBackbone
from losses import AdaFace, CurricularFace, FocalLoss
from augmentations import mixup_cutmix_tokenmix, get_randaugment_transform
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sam import SAM
from lookahead import Lookahead
from optimizer_gc import apply_gradient_centralization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ArcFaceTrainer:
    def __init__(self, dataloader, num_classes=100, mix_method='mixup', use_curricularface=False):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Backbone
        self.backbone = HybridBackbone(use_cbam=True).to(self.device)

        # Loss head selection
        if use_curricularface:
            self.head = CurricularFace(embedding_size=1024, num_classes=num_classes).to(self.device)
        else:
            self.head = AdaFace(embedding_size=1024, num_classes=num_classes).to(self.device)

        # Optimizer stack: SAM + Lookahead + Gradient Centralization (GC)
        base_optimizer = torch.optim.AdamW
        raw_optimizer = base_optimizer(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )
        raw_optimizer = apply_gradient_centralization(raw_optimizer)
        self.optimizer = Lookahead(SAM(raw_optimizer))

        # OneCycleLR with per-layer decay suggestion (manually apply lower LR to early layers if needed)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer.optimizer.base_optimizer,
            max_lr=1e-3,
            steps_per_epoch=len(dataloader),
            epochs=30
        )

        self.criterion = FocalLoss(alpha=1.0, gamma=2.0, smoothing=0.1)
        self.mix_method = mix_method
        self.augment = get_randaugment_transform()

    def train(self, epochs=30):
        self.backbone.train()
        self.head.train()

        for epoch in range(epochs):
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                mixed_x, y_a, y_b, lam = mixup_cutmix_tokenmix(images, labels, method=self.mix_method)

                # First step (SAM first forward)
                features = self.backbone(mixed_x)
                logits = self.head(features, y_a)
                loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                # Second step (SAM second forward)
                features = self.backbone(mixed_x)
                logits = self.head(features, y_a)
                loss_second = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
                loss_second.backward()
                self.optimizer.second_step(zero_grad=True)

                self.scheduler.step()

            logging.info(f"Epoch {epoch + 1}/{epochs} completed.")

if __name__ == "__main__":
    # Example with CIFAR-100
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    cifar_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(cifar_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    trainer = ArcFaceTrainer(dataloader, num_classes=100, mix_method='mixup', use_curricularface=True)
    trainer.train(epochs=30)

# fine_tune_distill.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from backbone import HybridBackbone
from losses import AdaFace, FocalLoss
from augmentations import mixup_cutmix_tokenmix, get_randaugment_transform
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sam import SAM
from lookahead import Lookahead
from optimizer_gc import apply_gradient_centralization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FineTuneDistillTrainer:
    def __init__(self, dataloader, num_classes=100, distill_weight=0.3):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.distill_weight = distill_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize backbone and head
        self.backbone = HybridBackbone(use_cbam=True).to(self.device)
        self.head = AdaFace(embedding_size=1024, num_classes=num_classes).to(self.device)

        # Optimizer stack: SAM-V2 + Lookahead + GC
        base_optimizer = torch.optim.AdamW
        raw_optimizer = base_optimizer(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=5e-6,
            weight_decay=1e-4
        )
        raw_optimizer = apply_gradient_centralization(raw_optimizer)
        self.optimizer = Lookahead(SAM(raw_optimizer))

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer.optimizer.base_optimizer,
            max_lr=1e-5,
            steps_per_epoch=len(dataloader),
            epochs=10
        )

        self.criterion = FocalLoss(alpha=1.0, gamma=2.0, smoothing=0.1)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        self.feature_mse = nn.MSELoss()

        # Augmentations
        self.augment = get_randaugment_transform()

    def train(self, epochs=10):
        self.backbone.train()
        self.head.train()

        for epoch in range(epochs):
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Teacher forward pass (no grad)
                with torch.no_grad():
                    teacher_features = self.backbone(images)
                    teacher_logits = self.head(teacher_features, labels)
                    teacher_soft = F.softmax(teacher_logits.detach() / 2.0, dim=1)

                # Mixup/CutMix/TokenMix applied
                mixed_x, y_a, y_b, lam = mixup_cutmix_tokenmix(images, labels, method='mixup')

                # Student first step (SAM first forward)
                student_features = self.backbone(mixed_x)
                student_logits = self.head(student_features, y_a)

                # Losses: hard loss + soft distillation + feature MSE distillation
                hard_loss = lam * self.criterion(student_logits, y_a) + (1 - lam) * self.criterion(student_logits, y_b)
                soft_loss = self.kldiv(F.log_softmax(student_logits / 2.0, dim=1), teacher_soft) * (2.0 ** 2)
                feature_loss = self.feature_mse(student_features, teacher_features)

                total_loss = (1 - self.distill_weight) * hard_loss + \
                             (self.distill_weight / 2) * soft_loss + \
                             (self.distill_weight / 2) * feature_loss

                total_loss.backward()
                self.optimizer.first_step(zero_grad=True)

                # Student second step (SAM second forward)
                student_features = self.backbone(mixed_x)
                student_logits = self.head(student_features, y_a)

                hard_loss_2 = lam * self.criterion(student_logits, y_a) + (1 - lam) * self.criterion(student_logits, y_b)
                soft_loss_2 = self.kldiv(F.log_softmax(student_logits / 2.0, dim=1), teacher_soft) * (2.0 ** 2)
                feature_loss_2 = self.feature_mse(student_features, teacher_features)

                total_loss_2 = (1 - self.distill_weight) * hard_loss_2 + \
                               (self.distill_weight / 2) * soft_loss_2 + \
                               (self.distill_weight / 2) * feature_loss_2

                total_loss_2.backward()
                self.optimizer.second_step(zero_grad=True)
                self.scheduler.step()

            logging.info(f"Epoch {epoch + 1}/{epochs} fine-tuning + distillation done.")


if __name__ == "__main__":
    # Example with CIFAR-100
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    cifar_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(cifar_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    trainer = FineTuneDistillTrainer(dataloader, num_classes=100)
    trainer.train(epochs=10)

# evaluate.py

import torch
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from backbone import HybridBackbone
from losses import AdaFace
from augmentations import mixup_cutmix_tokenmix, get_randaugment_transform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Evaluator:
    def __init__(self, dataloader, num_classes=100):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, weight_path):
        model = HybridBackbone(use_cbam=True).to(self.device)
        head = AdaFace(1024, self.num_classes).to(self.device)
        model.load_state_dict(torch.load(weight_path, map_location=self.device), strict=False)
        model.eval()
        head.eval()
        return model, head

    def evaluate(self, model, head, use_tta=False, use_ensemble=False, models_heads=None, ood_detection=False):
        y_true, y_pred, y_conf, energy_scores = [], [], [], []

        for images, labels in self.dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            if use_ensemble:
                preds = []
                for m, h in models_heads:
                    with torch.no_grad():
                        features = m(images)
                        logits = h(features, labels)
                        preds.append(F.softmax(logits, dim=1))
                avg_preds = torch.mean(torch.stack(preds), dim=0)
            elif use_tta:
                preds = []
                for _ in range(5):
                    aug_imgs, y_a, y_b, lam = mixup_cutmix_tokenmix(images, labels, method='mixup')
                    with torch.no_grad():
                        features = model(aug_imgs)
                        logits = head(features, y_a)
                        soft_logits = F.softmax(logits, dim=1)
                        preds.append(soft_logits)
                avg_preds = torch.mean(torch.stack(preds), dim=0)
            else:
                with torch.no_grad():
                    features = model(images)
                    logits = head(features, labels)
                    avg_preds = F.softmax(logits, dim=1)

            pred_labels = torch.argmax(avg_preds, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred_labels.cpu().numpy())
            y_conf.extend(torch.max(avg_preds, dim=1)[0].cpu().numpy())

            if ood_detection:
                energy = torch.logsumexp(logits, dim=1)
                energy_scores.extend(energy.cpu().numpy())

        self._print_report(y_true, y_pred)
        self._plot_confusion_matrix(y_true, y_pred)

        if ood_detection:
            self._plot_energy_histogram(energy_scores)

        return y_true, y_pred, y_conf

    def _print_report(self, y_true, y_pred):
        logging.info("\n" + classification_report(y_true, y_pred, zero_division=0))

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def _plot_energy_histogram(self, energy_scores):
        plt.figure(figsize=(10, 6))
        plt.hist(energy_scores, bins=100, color='purple', alpha=0.7)
        plt.title("Energy-based OOD Detection Score Distribution")
        plt.xlabel("Energy Score")
        plt.ylabel("Frequency")
        plt.show()


def calibrate_temperature(logits, labels):
    """Temperature Scaling"""
    temperature = torch.tensor(1.0, requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    logging.info(f"Optimal Temperature: {temperature.item():.4f}")
    return temperature.item()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    cifar_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(cifar_test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    evaluator = Evaluator(dataloader, num_classes=100)
    model, head = evaluator.load_model("best_model.pth")

    y_true, y_pred, y_conf = evaluator.evaluate(model, head, use_tta=True, ood_detection=True)

# utils.py

import torch
import logging
import os
from pathlib import Path
import torch.nn.functional as F

# ------------------------------
# Logger Setup
# ------------------------------
def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


# ------------------------------
# Checkpoint Utilities
# ------------------------------
def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    logging.info(f"✅ Checkpoint saved at {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info(f"🔄 Checkpoint loaded from {filename}")
    else:
        logging.warning(f"⚠️ No checkpoint found at {filename}")


# ------------------------------
# AMP (Automatic Mixed Precision) Wrapper
# ------------------------------
class AMPWrapper:
    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler()

    def backward(self, loss, optimizer):
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()


# ------------------------------
# Stochastic Weight Averaging (SWA) Utility
# ------------------------------
def apply_swa(model, swa_model, swa_start, step):
    if step >= swa_start:
        for swa_param, param in zip(swa_model.parameters(), model.parameters()):
            swa_param.data.mul_(0.99).add_(0.01 * param.data)


# ------------------------------
# Snapshot Ensembling Utility
# ------------------------------
def save_snapshot(model, epoch, folder="./snapshots"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    snapshot_file = os.path.join(folder, f"snapshot_epoch_{epoch}.pth")
    torch.save(model.state_dict(), snapshot_file)
    logging.info(f"📸 Snapshot saved at {snapshot_file}")


# ------------------------------
# Apply Gradient Centralization (GC)
# ------------------------------
def apply_gradient_centralization(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None and len(param.shape) > 1:
                param.grad.data.add_(-param.grad.data.mean(dim=tuple(range(1, len(param.shape))), keepdim=True))
    return optimizer


# ------------------------------
# Apply Token Merging (ToMe) Utility
# ------------------------------
def apply_token_merging(vit_model, ratio=0.5):
    try:
        import tome
    except ImportError:
        logging.error("ToMe library not found. Install: pip install git+https://github.com/GeorgeCazenavette/tome.git")
        return vit_model

    vit_model = tome.patch_vit(vit_model)
    vit_model.r = ratio
    logging.info(f"ToMe activated with ratio {ratio}")
    return vit_model


if __name__ == "__main__":
    Path("./logs").mkdir(parents=True, exist_ok=True)
    setup_logger("./logs/training.log")

    dummy_model = torch.nn.Linear(10, 2).cuda()
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())

    save_checkpoint({
        'state_dict': dummy_model.state_dict(),
        'optimizer': dummy_optimizer.state_dict()
    }, filename="test_checkpoint.pth")

    load_checkpoint(dummy_model, dummy_optimizer, filename="test_checkpoint.pth")

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

# train_supcon.py

import torch
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from backbone import HybridBackbone
from losses import SupConLoss
from augmentations import get_randaugment_transform
from torchvision import datasets, transforms
from sam import SAM
from lookahead import Lookahead
from optimizer_gc import apply_gradient_centralization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SupConPretrainer:
    def __init__(self, dataloader, num_classes=100, num_views=4):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.num_views = num_views
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HybridBackbone(use_cbam=True).to(self.device)
        self.loss_fn = SupConLoss(temperature=0.07)

        # Optimizer stack: SAM-V2 + Lookahead + GC
        base_optimizer = torch.optim.AdamW
        raw_optimizer = base_optimizer(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        raw_optimizer = apply_gradient_centralization(raw_optimizer)
        self.optimizer = Lookahead(SAM(raw_optimizer))

        self.augment = get_randaugment_transform()

    def multi_view_augment(self, x):
        """Generate multiple views per sample using RandAugment."""
        views = [self.augment(x) for _ in range(self.num_views)]
        return torch.stack(views)

    def train(self, steps=100):
        self.model.train()
        for step, (images, labels) in enumerate(self.dataloader):
            if step >= steps:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Create multiple views per image
            views = torch.cat([self.multi_view_augment(img.cpu()).to(self.device) for img in images], dim=0)
            expanded_labels = labels.repeat_interleave(self.num_views)

            # First forward (SAM first step)
            feats = self.model(views)
            loss = self.loss_fn(feats, expanded_labels)
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # Second forward (SAM second step)
            feats = self.model(views)
            loss2 = self.loss_fn(feats, expanded_labels)
            loss2.backward()
            self.optimizer.second_step(zero_grad=True)

            logging.info(f"Step {step}: SupCon Loss = {loss.item():.4f}")


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    cifar_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(cifar_train, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    trainer = SupConPretrainer(dataloader, num_classes=100)
    trainer.train(steps=100)


# run_pipeline.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
from train_arcface import ArcFaceTrainer
from fine_tune_distill import FineTuneDistillTrainer
from evaluate import Evaluator, calibrate_temperature
from utils import setup_logger, apply_swa, save_snapshot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_dataloader(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train_phase():
    train_loader, _ = get_dataloader()
    trainer = ArcFaceTrainer(train_loader, num_classes=100, mix_method='mixup', use_curricularface=True)
    trainer.train(epochs=30)


def fine_tune_phase():
    train_loader, _ = get_dataloader()
    distiller = FineTuneDistillTrainer(train_loader, num_classes=100)
    distiller.train(epochs=10)


def evaluate_phase():
    _, test_loader = get_dataloader()
    evaluator = Evaluator(test_loader, num_classes=100)

    # Load trained model
    model, head = evaluator.load_model("best_model.pth")

    # Evaluate with TTA, OOD detection, Calibration
    y_true, y_pred, y_conf = evaluator.evaluate(model, head, use_tta=True, ood_detection=True)

    # Optional: Temperature calibration on logits
    # (You can capture logits during evaluate if needed)
    # logits = capture_logits_during_evaluation()
    # temperature = calibrate_temperature(logits, torch.tensor(y_true).to(logits.device))


def swa_and_snapshot_phase():
    # Example pseudo-usage (replace with your models)
    dummy_model = torch.nn.Linear(10, 2).cuda()
    swa_model = torch.nn.Linear(10, 2).cuda()
    for step in range(100):
        # training_step(dummy_model)
        apply_swa(dummy_model, swa_model, swa_start=80, step=step)
        if step % 20 == 0:
            save_snapshot(dummy_model, epoch=step)


if __name__ == "__main__":
    setup_logger("./logs/full_pipeline.log")

    logging.info("==== TRAIN PHASE ====")
    train_phase()

    logging.info("==== FINE-TUNE + DISTILL PHASE ====")
    fine_tune_phase()

    logging.info("==== EVALUATE PHASE ====")
    evaluate_phase()

    logging.info("==== SWA + SNAPSHOT PHASE ====")
    swa_and_snapshot_phase()
