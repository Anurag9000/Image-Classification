import torch
import torch.nn.functional as F
import logging
import os
import csv
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from backbone import HybridBackbone
from losses import SupConLoss
from augmentations import get_randaugment_transform
from sam import SAM
from lookahead import Lookahead
from optimizer_gc import apply_gradient_centralization
from utils import save_snapshot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SupConPretrainer:
    def __init__(self, dataloader, num_classes=100, num_views=4, temperature=0.07,
                 lr=1e-3, log_csv="./logs/supcon_metrics.csv", snapshot_path="./snapshots_supcon/final.pth"):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.num_views = num_views
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HybridBackbone(use_cbam=True).to(self.device)
        self.loss_fn = SupConLoss(temperature=temperature)

        base_optimizer = torch.optim.AdamW
        raw_optimizer = base_optimizer(self.model.parameters(), lr=lr, weight_decay=1e-4)
        raw_optimizer = apply_gradient_centralization(raw_optimizer)
        self.optimizer = Lookahead(SAM(raw_optimizer))

        self.augment = get_randaugment_transform()
        self.log_csv = log_csv
        self.snapshot_path = snapshot_path

        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        with open(log_csv, 'w', newline='') as f:
            csv.writer(f).writerow(["step", "supcon_loss"])

    def multi_view_augment(self, x):
        return torch.stack([self.augment(x) for _ in range(self.num_views)])

    def train(self, steps=100):
        self.model.train()
        step_count = 0

        for step, (images, labels) in enumerate(self.dataloader):
            if step_count >= steps:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            views = torch.cat([self.multi_view_augment(img.cpu()).to(self.device) for img in images], dim=0)
            expanded_labels = labels.repeat_interleave(self.num_views)

            feats = self.model(views)
            loss = self.loss_fn(feats, expanded_labels)
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            feats = self.model(views)
            loss2 = self.loss_fn(feats, expanded_labels)
            loss2.backward()
            self.optimizer.second_step(zero_grad=True)

            step_count += 1
            logging.info(f"Step {step_count}: SupCon Loss = {loss.item():.4f}")

            with open(self.log_csv, 'a', newline='') as f:
                csv.writer(f).writerow([step_count, loss.item()])

        os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.snapshot_path)
        logging.info(f" SupCon pretrained weights saved to {self.snapshot_path}")


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    cifar_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(cifar_train, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    trainer = SupConPretrainer(dataloader, num_classes=100)
    trainer.train(steps=100)
