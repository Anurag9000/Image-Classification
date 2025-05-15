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
