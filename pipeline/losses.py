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

    def forward(self, embeddings, labels=None):
        # Fix: Force float32 for metric learning to prevent NaN in AMP
        embeddings = embeddings.float()
        if hasattr(self, "weight"):
             self.weight.data = self.weight.data.float() # Ensure weight is float32? usually auto-handled 
        
        # Manually casting weight to float for calculation
        weight = self.weight.float()

        cosine = F.linear(F.normalize(embeddings), F.normalize(weight))
        if labels is None:
            return cosine * self.s

        norm = torch.norm(embeddings, dim=1, keepdim=True).clamp(min=self.eps)

        with torch.no_grad():
            batch_mean = norm.mean()
            batch_std = norm.std()
            self.batch_mean = (1 - self.momentum) * self.batch_mean + self.momentum * batch_mean
            self.batch_std = (1 - self.momentum) * self.batch_std + self.momentum * batch_std

        margin_scaler = ((norm - self.batch_mean) / (self.batch_std + self.eps)).clamp(-1, 1)
        adaptive_m = self.m * (1 + margin_scaler)

        # acos numeric stability
        theta = torch.acos(cosine.clamp(-1 + self.eps, 1 - self.eps))
        target_logit = torch.cos(theta + adaptive_m.view(-1, 1))

        one_hot = torch.zeros_like(cosine)
        if labels.max() >= one_hot.size(1) or labels.min() < 0:
             # Gracefully handle invalid labels by ignoring them (masking) or raising clearer error
             # Raising error is better for debugging training data issues
             raise ValueError(f"Label index out of range in AdaFace: max {labels.max()} vs valid {one_hot.size(1)}")
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

    def forward(self, embeddings, labels=None):
        # Fix: Force float32 for metric learning to prevent NaN in AMP
        embeddings = embeddings.float()
        weight = self.weight.float()

        cosine = F.linear(F.normalize(embeddings), F.normalize(weight))
        if labels is None:
            return cosine * self.s

        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_cos = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        if labels.max() >= one_hot.size(1) or labels.min() < 0:
             raise ValueError(f"Label index out of range in CurricularFace: max {labels.max()} vs valid {one_hot.size(1)}")
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        hard_mask = cosine > target_cos
        if hard_mask.sum() > 0:
            self.t.data = 0.01 * cosine[hard_mask].mean().detach() + 0.99 * self.t.data
        else:
            # Keep t as is if no hard samples
            pass

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
        # Fix: Cast to float32 to prevent exp() overflow in float16 (AMP)
        # exp(1/0.07) ~ 1.4M > 65k (float16 max)
        features = features.float()

        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Numeric stability: sub max for exp
        similarity_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - similarity_max.detach()

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
    try:
        if torch.cuda.is_available():
            dummy_feats = torch.randn(8, 1024).cuda()
            dummy_labels = torch.randint(0, 100, (8,)).cuda()

            adaface = AdaFace(1024, 100).cuda()
            logits = adaface(dummy_feats, dummy_labels)
            # print("AdaFace logits:", logits.shape)
        print("Losses module loaded successfully.")
    except Exception as e:
        print(f"Losses smoke test failed: {e}")
