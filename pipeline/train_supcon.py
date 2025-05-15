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
