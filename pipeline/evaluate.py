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
