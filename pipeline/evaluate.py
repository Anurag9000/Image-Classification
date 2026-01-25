from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from .backbone import BackboneConfig, HybridBackbone
from .losses import AdaFace

LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    result_dir: str = "./eval_results"
    tta_runs: int = 5
    tta_mix_method: str = "mixup"
    topk: Sequence[int] = (1, 5)
    compute_calibration: bool = True
    ood_detection: bool = False
    backbone: dict = field(default_factory=dict)
    compute_mahalanobis: bool = False
    compute_vim: bool = False


class Evaluator:
    def __init__(self, dataloader, num_classes: int = 100, config: Optional[EvaluationConfig] = None):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = config or EvaluationConfig()
        self.result_dir = self.cfg.result_dir
        os.makedirs(self.result_dir, exist_ok=True)

    def load_model(
        self,
        backbone_path: str,
        head_path: Optional[str] = None,
        strict: bool = False,
    ):
        backbone_cfg = BackboneConfig(**self.cfg.backbone)
        model = HybridBackbone(backbone_cfg).to(self.device)
        head = AdaFace(backbone_cfg.fusion_dim, self.num_classes).to(self.device)

        backbone_state = torch.load(backbone_path, map_location=self.device)
        if isinstance(backbone_state, dict) and "state_dict" in backbone_state:
            backbone_state = backbone_state["state_dict"]
        if "backbone" in backbone_state:
            model.load_state_dict(backbone_state["backbone"], strict=strict)
            if head_path is None and "head" in backbone_state:
                head.load_state_dict(backbone_state["head"], strict=strict)
        else:
            model.load_state_dict(backbone_state, strict=strict)

        if head_path:
            head_state = torch.load(head_path, map_location=self.device)
            if isinstance(head_state, dict) and "state_dict" in head_state:
                head_state = head_state["state_dict"]
            head.load_state_dict(head_state, strict=strict)

        model.eval()
        head.eval()
        return model, head

    def _apply_tta(self, images: torch.Tensor, labels: torch.Tensor, model, head) -> Tuple[torch.Tensor, torch.Tensor]:
        preds: List[torch.Tensor] = []
        logits_collector: List[torch.Tensor] = []

        tta_ops = [
            lambda x: x,
            lambda x: torch.flip(x, dims=[3]),
            lambda x: torch.flip(x, dims=[2]),
            lambda x: torch.rot90(x, k=1, dims=(2, 3)),
            lambda x: torch.rot90(x, k=3, dims=(2, 3)),
        ]

        for idx in range(self.cfg.tta_runs):
            op = tta_ops[idx % len(tta_ops)]
            aug_imgs = op(images.clone())
            with torch.no_grad():
                features = model(aug_imgs)
                logits = head(features, labels=None)
                logits_collector.append(logits)
                preds.append(F.softmax(logits, dim=1))

        avg_preds = torch.mean(torch.stack(preds), dim=0)
        avg_logits = torch.mean(torch.stack(logits_collector), dim=0)
        return avg_preds, avg_logits

    def _compute_topk(self, probs: torch.Tensor, labels: torch.Tensor, ks: Iterable[int]) -> dict:
        metrics = {}
        for k in ks:
            real_k = min(k, probs.size(1))
            if real_k < 1:
                continue
            topk = torch.topk(probs, k=real_k, dim=1).indices
            correct = topk.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()
            metrics[f"top{k}"] = correct
        return metrics

    def _expected_calibration_error(self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
            if not np.any(mask):
                continue
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
        return float(ece)

    def evaluate(self, model, head, models_heads: Optional[List[Tuple[torch.nn.Module, torch.nn.Module]]] = None):
        y_true, y_pred, y_conf, energy_scores = [], [], [], []
        logits_list, probs_list = [], []
        feature_bank = [] if (self.cfg.compute_mahalanobis or self.cfg.compute_vim) else None

        for images, labels in self.dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if models_heads:
                preds = []
                logits_stack = []
                for m, h in models_heads:
                    with torch.no_grad():
                        features = m(images)
                        logits = h(features, labels=None)
                        logits_stack.append(logits)
                        preds.append(F.softmax(logits, dim=1))
                avg_logits = torch.mean(torch.stack(logits_stack), dim=0)
                avg_preds = torch.mean(torch.stack(preds), dim=0)
            else:
                avg_preds, avg_logits = (
                    self._apply_tta(images, labels, model, head)
                    if self.cfg.tta_runs > 1
                    else self._forward_single(model, head, images, labels)
                )

            probs_list.append(avg_preds.cpu())
            logits_list.append(avg_logits.cpu())

            pred_labels = torch.argmax(avg_preds, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred_labels.cpu().numpy())
            y_conf.extend(torch.max(avg_preds, dim=1)[0].cpu().numpy())

            if self.cfg.ood_detection:
                energy = torch.logsumexp(avg_logits, dim=1)
                energy_scores.extend(energy.cpu().numpy())

            if feature_bank is not None:
                with torch.no_grad():
                    base_features = model(images)
                feature_bank.append(base_features.detach().cpu())

        probs_concat = torch.cat(probs_list)
        logits_concat = torch.cat(logits_list)
        metrics = self._summarize(y_true, y_pred, y_conf, probs_concat, logits_concat)

        if feature_bank:
            features = torch.cat(feature_bank)
            labels_tensor = torch.tensor(y_true)
            self._update_ood_metrics(metrics, features, labels_tensor)

        if self.cfg.ood_detection and energy_scores:
            self._plot_energy_histogram(energy_scores)

        return metrics

    def _forward_single(self, model, head, images, labels):
        with torch.no_grad():
            features = model(images)
            logits = head(features, labels=None)
            probs = F.softmax(logits, dim=1)
        return probs, logits

    def _summarize(self, y_true, y_pred, y_conf, probs, logits):
        self._save_classification_report(y_true, y_pred)
        self._plot_confusion_matrix(y_true, y_pred)
        self._save_predictions_csv(y_true, y_pred, y_conf)

        metrics = {
            "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        }
        topk_metrics = self._compute_topk(probs, torch.tensor(y_true, dtype=torch.long), self.cfg.topk)
        metrics.update({f"top_{k}": v for k, v in topk_metrics.items()})

        if self.cfg.compute_calibration:
            confidences = probs.max(dim=1).values.numpy()
            accuracies = (np.array(y_true) == np.array(y_pred)).astype(float)
            metrics["ece"] = self._expected_calibration_error(confidences, accuracies)

        metrics_path = os.path.join(self.result_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        LOGGER.info("Evaluation metrics saved to %s", metrics_path)
        return metrics

    def _save_predictions_csv(self, y_true, y_pred, confidences):
        csv_path = os.path.join(self.result_dir, "predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["y_true", "y_pred", "confidence"])
            writer.writerows(zip(y_true, y_pred, confidences))
        LOGGER.info("Predictions saved to %s", csv_path)

    def _save_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        report_path = os.path.join(self.result_dir, "classification_report.csv")
        with open(report_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "precision", "recall", "f1-score", "support"])
            for cls, metrics in report.items():
                if isinstance(metrics, dict):
                    writer.writerow([
                        cls,
                        metrics.get("precision", 0),
                        metrics.get("recall", 0),
                        metrics.get("f1-score", 0),
                        metrics.get("support", 0),
                    ])
        LOGGER.info("Classification report saved to %s", report_path)

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, "confusion_matrix.png"))
        plt.close()
        LOGGER.info("Confusion matrix saved.")

    def _plot_energy_histogram(self, energy_scores):
        plt.figure(figsize=(10, 6))
        plt.hist(energy_scores, bins=100, color="purple", alpha=0.7)
        plt.title("Energy-based OOD Detection Score Distribution")
        plt.xlabel("Energy Score")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, "ood_energy_distribution.png"))
        plt.close()
        LOGGER.info("Energy histogram saved.")

    def _update_ood_metrics(self, metrics, features: torch.Tensor, labels: torch.Tensor):
        feats = features.float()
        labels = labels.long()
        if self.cfg.compute_mahalanobis:
            class_means = {}
            centered_list = []
            for c in range(self.num_classes):
                mask = labels == c
                if mask.any():
                    class_feat = feats[mask]
                    mean = class_feat.mean(dim=0)
                    class_means[c] = mean
                    centered_list.append(class_feat - mean)
            if centered_list:
                centered = torch.cat(centered_list, dim=0)
                cov = (centered.T @ centered) / max(centered.size(0) - 1, 1)
                cov += torch.eye(cov.size(0)) * 1e-6
                inv_cov = torch.linalg.pinv(cov)
                maha_scores = []
                for idx in range(feats.size(0)):
                    label = labels[idx].item()
                    mean = class_means.get(label)
                    if mean is None:
                        continue
                    diff = (feats[idx] - mean).unsqueeze(0)
                    score = torch.matmul(torch.matmul(diff, inv_cov), diff.t()).item()
                    maha_scores.append(score)
                if maha_scores:
                    metrics["mahalanobis_mean"] = float(np.mean(maha_scores))
                    metrics["mahalanobis_std"] = float(np.std(maha_scores))
                    self._save_vector_csv("mahalanobis_scores.csv", maha_scores)

        if self.cfg.compute_vim:
            try:
                centered = feats - feats.mean(dim=0, keepdim=True)
                # Adaptive rank selection for PCA
                q = min(64, centered.size(1), centered.size(0) - 1)
                if q > 0:
                    _, _, v = torch.pca_lowrank(centered, q=q)
                    residual = centered - centered @ v @ v.t()
                    vim_scores = residual.norm(dim=1).cpu().numpy()
                    metrics["vim_score_mean"] = float(np.mean(vim_scores))
                    metrics["vim_score_std"] = float(np.std(vim_scores))
                    self._save_vector_csv("vim_scores.csv", vim_scores)
                else:
                    LOGGER.warning("Not enough samples/features for ViM PCA.")
            except Exception as exc: # Catch all PCA-related errors
                LOGGER.warning("ViM computation failed: %s", exc)

    def _save_vector_csv(self, filename: str, values):
        path = os.path.join(self.result_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["score"])
            for val in values:
                writer.writerow([float(val)])
        LOGGER.info("Saved %s", path)





if __name__ == "__main__":
    # Example usage for testing
    from torchvision import transforms
    from .files_dataset import create_data_loader
    
    # Simple test run if executed directly
    logging.basicConfig(level=logging.INFO)
    try:
         _, _, dataloader = create_data_loader(
            root_dirs=["./data/Dataset_Final"],
            batch_size=32,
            test_split=1.0 # Use full for testing
        )
         # evaluator = Evaluator(dataloader, num_classes=100)
         # model, head = evaluator.load_model("snapshots/snapshot_epoch_30.pth")
         # evaluator.evaluate(model, head)
         LOGGER.info("Evaluator imported successfully.")
    except Exception as e:
         LOGGER.warning(f"Could not run dummy evaluation: {e}")

