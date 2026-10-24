import pytest
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.registry import ModelRegistry
from training.loss import get_loss_fn
from training.trainer import Trainer

# Force model registration
import models

def test_trainer_weight_updates():
    """Verify that Trainer optimization steps successfully update model weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Create a dummy model
    model = ModelRegistry.create(
        "cnn",
        input_channels=12,
        num_classes=5,
        filters=[16],  # Very small model for speed
        fc_dim=16,
        pooling="avg"
    ).to(device)
    
    # Keep track of initial weights
    initial_params = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

    # 2. Setup mock data
    x = torch.randn(8, 12, 1000)
    y = torch.randint(0, 2, (8, 5)).float()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)

    # 3. Setup optimizer, loss, configurations
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = get_loss_fn("focal")
    
    # Mini Hydra-like config mock
    class EnvConfig:
        mixed_precision = False
    
    class ESConfig:
        patience = 5
        monitor = "val_loss"
        mode = "min"
        
    class CPConfig:
        save_dir = "./tmp_ckpt"
        monitor = "val_loss"
        mode = "min"
        
    class SchedulerConfig:
        name = "none"

    class TrainingConfig:
        epochs = 1
        batch_size = 4
        grad_accumulation_steps = 1
        grad_clip_norm = 1.0
        early_stopping = ESConfig()
        checkpoint = CPConfig()
        scheduler = {"name": "none"}
        
    class ConfigMock:
        env = EnvConfig()
        training = TrainingConfig()

    cfg = ConfigMock()

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        run_dir="./tmp_run"
    )
    
    # Run a single epoch training
    avg_loss = trainer.train_epoch(1)
    
    # 5. Assertions
    assert avg_loss > 0.0
    
    # Check that weights have changed
    updated_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    any_param_updated = False
    for name in initial_params.keys():
        if not torch.equal(initial_params[name], updated_params[name]):
            any_param_updated = True
            break
            
    assert any_param_updated, "Model parameters were not modified during optimization step."
    
    # Cleanup dummy run folders
    import shutil
    if os.path.exists("./tmp_run"):
        shutil.rmtree("./tmp_run")
