import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import csv
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score

from backbone import HybridBackbone
from losses import AdaFace, FocalLoss, EvidentialLoss
from augmentations import mixup_cutmix_tokenmix, get_randaugment_transform
from sam import SAM
from lookahead import Lookahead
from optimizer_gc import apply_gradient_centralization
from utils import save_snapshot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FineTuneDistillTrainer:
    def __init__(self, dataloader, num_classes=100, distill_weight=0.3,
                 mix_method='mixup', use_evidential=False, lr=5e-6,
                 snapshot_dir="./snapshots_distill", log_csv="./logs/distill_metrics.csv"):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.distill_weight = distill_weight
        self.mix_method = mix_method
        self.use_evidential = use_evidential
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = HybridBackbone(use_cbam=True).to(self.device)
        self.head = AdaFace(embedding_size=1024, num_classes=num_classes).to(self.device)

        base_optimizer = torch.optim.AdamW
        raw_optimizer = base_optimizer(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=1e-4
        )
        raw_optimizer = apply_gradient_centralization(raw_optimizer)
        self.optimizer = Lookahead(SAM(raw_optimizer))

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer.optimizer.base_optimizer,
            max_lr=lr * 2,
            steps_per_epoch=len(dataloader),
            epochs=10
        )

        if use_evidential:
            self.criterion = EvidentialLoss(num_classes=num_classes)
        else:
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0, smoothing=0.1)

        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        self.feature_mse = nn.MSELoss()
        self.augment = get_randaugment_transform()

        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        self.log_csv = log_csv
        self.snapshot_dir = snapshot_dir
        with open(self.log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "train_f1"])

    def train(self, epochs=10):
        self.backbone.train()
        self.head.train()

        for epoch in range(epochs):
            epoch_losses = []
            preds_all, labels_all = [], []

            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    teacher_features = self.backbone(images)
                    teacher_logits = self.head(teacher_features, labels)
                    teacher_soft = F.softmax(teacher_logits.detach() / 2.0, dim=1)

                mixed_x, y_a, y_b, lam = mixup_cutmix_tokenmix(images, labels, method=self.mix_method)

                # First step
                student_features = self.backbone(mixed_x)
                student_logits = self.head(student_features, y_a)

                hard_loss = lam * self.criterion(student_logits, y_a) + (1 - lam) * self.criterion(student_logits, y_b)
                soft_loss = self.kldiv(F.log_softmax(student_logits / 2.0, dim=1), teacher_soft) * (2.0 ** 2)
                feature_loss = self.feature_mse(student_features, teacher_features)

                total_loss = (1 - self.distill_weight) * hard_loss + \
                             (self.distill_weight / 2) * soft_loss + \
                             (self.distill_weight / 2) * feature_loss

                total_loss.backward()
                self.optimizer.first_step(zero_grad=True)

                # Second step
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

                epoch_losses.append(total_loss_2.item())
                preds_all.extend(torch.argmax(student_logits, dim=1).cpu().numpy())
                labels_all.extend(y_a.cpu().numpy())

            acc = accuracy_score(labels_all, preds_all)
            f1 = f1_score(labels_all, preds_all, average='macro')
            avg_loss = sum(epoch_losses) / len(epoch_losses)

            logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

            with open(self.log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_loss, acc, f1])

            save_snapshot(self.backbone, epoch=epoch+1, folder=self.snapshot_dir)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    cifar_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(cifar_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    trainer = FineTuneDistillTrainer(
        dataloader,
        num_classes=100,
        distill_weight=0.3,
        mix_method='mixup',
        use_evidential=False
    )
    trainer.train(epochs=10)
