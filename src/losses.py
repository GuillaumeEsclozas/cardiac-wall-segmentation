"""
Loss functions for cardiac segmentation.

DiceCE (CE=0.3, Dice=0.7) — proven best loss across all experiments.
DeepSupLoss wraps any base loss to add deep supervision from auxiliary outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """DiceCE 30/70 — proven best loss across all experiments."""

    def __init__(self, num_classes=4, ce_weight=0.3, dice_weight=0.7,
                 class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        weight = class_weights if class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        inter = (probs * targets_oh).sum(dims)
        union = probs.sum(dims) + targets_oh.sum(dims)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class DeepSupLoss(nn.Module):
    """Wraps any base loss to add deep supervision from auxiliary outputs."""

    def __init__(self, base_criterion, aux_weights=(0.1, 0.1, 0.1, 0.1)):
        super().__init__()
        self.base_criterion = base_criterion
        self.aux_weights = aux_weights

    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            main_out, aux_outputs = outputs
            loss = self.base_criterion(main_out, target)
            for i, aux in enumerate(aux_outputs):
                if i < len(self.aux_weights):
                    loss = loss + self.aux_weights[i] * self.base_criterion(aux, target)
            return loss
        return self.base_criterion(outputs, target)
