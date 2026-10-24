import torch
import math
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class LRFinder:
    """Range test to find optimal learning rate.

    Gradually increases the learning rate from a low value to a high value
    over a mock training run, recording loss at each step to plot and inspect.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Save model and optimizer states to restore them later
        self.model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.optimizer_state = {k: v for k, v in optimizer.state_dict().items()}

    def restore(self) -> None:
        """Restores model and optimizer states to prevent range test side effects."""
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        logger.info("Model and optimizer states successfully restored after LR Range Test.")

    def range_test(
        self,
        train_loader: torch.utils.data.DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        diverge_th: float = 4.0
    ) -> Tuple[List[float], List[float]]:
        """Executes the learning rate sweep.

        Args:
            train_loader: Dataloader supplying train batches.
            start_lr: Starting learning rate parameter.
            end_lr: Ceiling learning rate limit.
            num_iter: Steps to run the test.
            diverge_th: Divergence threshold (stop if loss > diverge_th * best_loss).

        Returns:
            Tuple[List[float], List[float]]: Learning rates and corresponding losses.
        """
        self.model.train()
        lrs = []
        losses = []
        
        # Calculate multiplier
        mult = (end_lr / start_lr) ** (1.0 / num_iter)
        lr = start_lr
        
        # Set initial learning rate
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
            
        best_loss = float("inf")
        avg_loss = 0.0
        smoothed_loss = 0.0
        beta = 0.98  # Smoothing factor
        
        iter_idx = 0
        loader_iter = iter(train_loader)
        
        while iter_idx < num_iter:
            try:
                inputs, targets = next(loader_iter)
            except StopIteration:
                # Wrap dataloader if exhausted
                loader_iter = iter(train_loader)
                inputs, targets = next(loader_iter)
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Stop if loss explodes to NaN
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at step {iter_idx}. Stopping LR finder.")
                break
                
            # Track running average loss
            iter_idx += 1
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** iter_idx)
            
            # Save stats
            lrs.append(lr)
            losses.append(smoothed_loss)
            
            # Check divergence
            if iter_idx > 1 and smoothed_loss > diverge_th * best_loss:
                logger.info(f"Loss diverged ({smoothed_loss:.4f} > {diverge_th * best_loss:.4f}). Stopping sweep.")
                break
                
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            # Exponentially increase learning rate
            lr *= mult
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
                
        # Reset model states
        self.restore()
        return lrs, losses
