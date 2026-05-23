from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import has_valid_label


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pt = torch.exp(-F.cross_entropy(input, target, reduction="none"))
        ce = F.cross_entropy(
            input,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        return ((1.0 - pt) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=-1)
        device = features.device
        num_rows = features.shape[0]
        if num_rows < 2:
            return features.new_zeros(())

        sim = features @ features.T / self.temperature
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag_mask = ~torch.eye(num_rows, dtype=torch.bool, device=device)
        pos_mask = label_mask & diag_mask

        if pos_mask.sum() == 0:
            return features.new_zeros(())

        logits_max = sim.max(dim=1, keepdim=True).values.detach()
        logits = sim - logits_max
        exp_logits = torch.exp(logits) * diag_mask.float()
        log_prob = logits - \
            torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        num_pos = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask.float() *
                         log_prob).sum(dim=1) / (num_pos + 1e-12)
        has_pos = num_pos > 0
        return -mean_log_prob[has_pos].mean()


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0

    def __call__(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_train_criterion(train_df, task_config: dict, num_classes: int, args, device: str):
    if task_config.get("multi_label"):
        pos_counts = np.zeros(num_classes, dtype=np.float64)
        labeled_rows = 0
        for value in train_df["class_id"].values:
            if not has_valid_label(value):
                continue
            label_arr = np.asarray(value, dtype=np.float32)
            if label_arr.shape[0] != num_classes:
                continue
            pos_counts += label_arr
            labeled_rows += 1

        neg_counts = np.maximum(labeled_rows - pos_counts, 0.0)
        pos_weight = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
        pos_weight = torch.tensor(
            pos_weight, dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    labeled = train_df[train_df["class_id"] != -1]
    label_counts = Counter(labeled["class_id"].values)
    total = sum(label_counts.values())
    class_weights = torch.zeros(num_classes)
    for class_id, count in label_counts.items():
        if class_id >= 0:
            class_weights[class_id] = total / (num_classes * count)
    class_weights = class_weights.to(device)

    return FocalLoss(
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        weight=class_weights,
    )


def build_eval_criterion(task_config: dict, args):
    if task_config.get("multi_label"):
        return nn.BCEWithLogitsLoss()
    return FocalLoss(gamma=args.focal_gamma)
