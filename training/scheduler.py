import torch
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LinearLR,
    SequentialLR
)
from typing import Any, Dict

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    epochs: int,
    steps_per_epoch: int
) -> Any:
    """Instantiates a learning rate scheduler according to configurations.

    Args:
        optimizer: Optimization object.
        cfg: Scheduler configuration dict.
        epochs: Total epochs to train.
        steps_per_epoch: Number of batches in one training epoch.

    Returns:
        Scheduler instance (e.g. SequentialLR, ReduceLROnPlateau, OneCycleLR)
    """
    name = cfg.get("name", "none").lower()
    
    if name == "cosine":
        warmup_epochs = cfg.get("warmup_epochs", 5)
        if warmup_epochs > 0 and epochs > warmup_epochs:
            # Linear warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            # Cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=(epochs - warmup_epochs),
                eta_min=1e-6
            )
            # Chain them sequentially
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            return scheduler
        else:
            return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            
    elif name == "onecycle":
        max_lr = cfg.get("max_lr", 3e-3)
        pct_start = cfg.get("pct_start", 0.3)
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy="cos"
        )
        
    elif name == "plateau":
        factor = cfg.get("factor", 0.1)
        patience = cfg.get("patience", 5)
        return ReduceLROnPlateau(
            optimizer,
            mode="max",  # Assuming we monitor a metric like validation F1
            factor=factor,
            patience=patience,
            verbose=True
        )
        
    return None
