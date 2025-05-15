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
