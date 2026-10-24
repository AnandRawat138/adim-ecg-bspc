import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    """Multi-label Binary Focal Loss.

    Reduces loss weights for easy-to-classify samples, forcing focus on hard cases.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SoftmaxFocalLoss(nn.Module):
    """Single-label (softmax) focal loss with label smoothing.

    loss = -alpha * sum_c q_c * (1 - p_c)^gamma * log p_c, where q is the smoothed one-hot target.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.shape[-1]
        log_p = F.log_softmax(inputs.float(), dim=-1)
        p = log_p.exp()
        q = F.one_hot(targets, num_classes).float()
        q = q * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
        loss = -(q * (1.0 - p) ** self.gamma * log_p).sum(dim=-1)
        if self.alpha >= 0:
            loss = self.alpha * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Dice Loss adapted for multi-label binary ECG predictions."""

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        
        # Calculate intersection and union along batches for each class
        intersection = (inputs * targets).sum(dim=0)
        union = inputs.sum(dim=0) + targets.sum(dim=0)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class LabelSmoothingBCELoss(nn.Module):
    """Binary Cross Entropy with manually applied label smoothing for multi-label tasks."""

    def __init__(self, smoothing: float = 0.1, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply smoothing: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            inputs, smoothed_targets, pos_weight=self.pos_weight, reduction=self.reduction
        )


def get_loss_fn(
    name: str,
    label_smoothing: float = 0.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: Optional[torch.Tensor] = None
) -> nn.Module:
    """Factory to retrieve loss functions.

    Args:
        name: Name of loss function ('cross_entropy', 'focal', 'dice', 'label_smoothing')
        label_smoothing: Amount of smoothing.
        focal_alpha: Alpha param for focal.
        focal_gamma: Gamma param for focal.
        pos_weight: Optional positive weights for BCE.

    Returns:
        nn.Module: Loss function module.
    """
    name = name.lower()
    if name == "softmax_focal":
        return SoftmaxFocalLoss(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif name == "dice":
        return DiceLoss()
    elif name == "label_smoothing" or (name == "cross_entropy" and label_smoothing > 0.0):
        return LabelSmoothingBCELoss(smoothing=label_smoothing, pos_weight=pos_weight)
    else:
        # Default standard multi-label BCE
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
