import pytest
import torch
from training.loss import get_loss_fn, FocalLoss, DiceLoss, LabelSmoothingBCELoss

def test_focal_loss_forward():
    inputs = torch.randn(4, 5)
    targets = torch.randint(0, 2, (4, 5)).float()
    
    # Instantiate FocalLoss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(inputs, targets)
    
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_dice_loss_forward():
    inputs = torch.randn(4, 5)
    targets = torch.randint(0, 2, (4, 5)).float()
    
    criterion = DiceLoss()
    loss = criterion(inputs, targets)
    
    assert loss.ndim == 0
    assert 0.0 <= loss.item() <= 1.0


def test_label_smoothing_bce_loss():
    inputs = torch.randn(4, 5)
    targets = torch.randint(0, 2, (4, 5)).float()
    
    criterion = LabelSmoothingBCELoss(smoothing=0.1)
    loss = criterion(inputs, targets)
    
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_get_loss_fn():
    focal = get_loss_fn("focal")
    assert isinstance(focal, FocalLoss)
    
    dice = get_loss_fn("dice")
    assert isinstance(dice, DiceLoss)
    
    smoothing = get_loss_fn("label_smoothing", label_smoothing=0.1)
    assert isinstance(smoothing, LabelSmoothingBCELoss)
    
    default_bce = get_loss_fn("cross_entropy")
    assert isinstance(default_bce, torch.nn.BCEWithLogitsLoss)
