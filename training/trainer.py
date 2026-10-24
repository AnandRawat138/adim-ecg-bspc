import os
import json
import csv
import logging
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple

from training.loss import get_loss_fn
from training.scheduler import get_scheduler
from utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

class CSVLogger:
    """Utility to log epoch statistics into a CSV table file."""

    def __init__(self, log_path: str, fieldnames: list) -> None:
        self.log_path = log_path
        self.fieldnames = fieldnames
        
        # Write header if file does not exist
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, metrics: dict) -> None:
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: v for k, v in metrics.items() if k in self.fieldnames})


class Trainer:
    """Main training orchestrator for the ECG framework.

    Supports AMP (Mixed Precision), early stopping, gradient accumulation,
    seeding, and isolated tensorboard/CSV metric storage.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Dict[str, Any],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device,
        run_dir: str
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.run_dir = run_dir

        self.epochs = cfg.training.epochs
        self.batch_size = cfg.training.batch_size
        
        if hasattr(cfg.training, "get"):
            self.grad_accum_steps = cfg.training.get("grad_accumulation_steps", 1)
            self.grad_clip_norm = cfg.training.get("grad_clip_norm", 1.0)
        else:
            self.grad_accum_steps = getattr(cfg.training, "grad_accumulation_steps", 1)
            self.grad_clip_norm = getattr(cfg.training, "grad_clip_norm", 1.0)
            
        self.mixed_precision = cfg.env.mixed_precision
        
        # Setup paths
        self.weights_dir = os.path.join(run_dir, "weights")
        self.logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Scheduler
        self.scheduler = get_scheduler(
            optimizer, 
            cfg.training.scheduler, 
            self.epochs, 
            len(train_loader)
        )

        # Loggers
        self.tb_writer = SummaryWriter(log_dir=self.logs_dir)
        self.csv_logger = CSVLogger(
            log_path=os.path.join(run_dir, "history.csv"),
            fieldnames=["epoch", "train_loss", "val_loss", "val_f1", "lr"]
        )

        # Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # Early Stopping state
        self.es_patience = cfg.training.early_stopping.patience
        self.es_monitor = cfg.training.early_stopping.monitor
        self.es_mode = cfg.training.early_stopping.mode
        self.es_counter = 0
        self.best_metric = -float("inf") if self.es_mode == "max" else float("inf")

        self.start_epoch = 1

    def train_epoch(self, epoch: int) -> float:
        """Runs a single epoch of training.

        Args:
            epoch: Current epoch number.

        Returns:
            float: Average training loss.
        """
        self.model.train()
        epoch_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{self.epochs} [Train]"
        )

        for batch_idx, (inputs, targets) in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Autocast Forward Pass
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps

            # Backward Pass
            self.scaler.scale(loss).backward()

            # Step optimizer at accumulation frequency
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Step step-level schedulers (like OneCycleLR)
                if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and not isinstance(self.scheduler, torch.optim.lr_scheduler.SequentialLR):
                    if hasattr(self.scheduler, "step"):
                        self.scheduler.step()

            epoch_loss += loss.item() * self.grad_accum_steps
            pbar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}"})

        # Step epoch-level schedulers (like CosineAnnealing or SequentialLR)
        if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.SequentialLR):
            self.scheduler.step()

        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Runs the validation split to compute loss and thresholded macro-F1.

        Returns:
            Tuple[float, float]: val_loss, val_f1
        """
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            val_loss += loss.item()
            
            # Predict probabilities (softmax for single-label integer targets, sigmoid for multi-hot)
            if targets.dim() == 1:
                probs = torch.softmax(outputs.float(), dim=-1).cpu().numpy()
            else:
                probs = torch.sigmoid(outputs).float().cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

        avg_loss = val_loss / len(self.val_loader)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        if all_targets.ndim == 1:
            # Single-label: macro-F1 of argmax predictions
            from sklearn.metrics import f1_score
            macro_f1 = float(f1_score(all_targets, all_preds.argmax(axis=1), average="macro",
                                      labels=list(range(all_preds.shape[1])), zero_division=0))
            return avg_loss, macro_f1

        # Compute a basic validation macro-F1 (using standard 0.5 threshold)
        # Detailed multi-threshold evaluations are computed in evaluation/metrics.py
        preds_bin = (all_preds > 0.5).astype(float)
        
        # Calculate macro F1
        f1_scores = []
        for i in range(all_targets.shape[1]):
            tp = np.sum((all_targets[:, i] == 1) & (preds_bin[:, i] == 1))
            fp = np.sum((all_targets[:, i] == 0) & (preds_bin[:, i] == 1))
            fn = np.sum((all_targets[:, i] == 1) & (preds_bin[:, i] == 0))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_scores.append(f1)
            
        macro_f1 = float(np.mean(f1_scores))
        return avg_loss, macro_f1

    def fit(self) -> None:
        """Launches the complete training/validation sequence."""
        logger.info("Beginning model training sequence.")
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Fetch current LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Train
            train_loss = self.train_epoch(epoch)
            # Validate
            val_loss, val_f1 = self.validate()
            
            # Step Plateau scheduler if applicable
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_f1)

            # Logs metrics to terminal
            logger.info(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Log to TensorBoard
            self.tb_writer.add_scalar("Loss/Train", train_loss, epoch)
            self.tb_writer.add_scalar("Loss/Val", val_loss, epoch)
            self.tb_writer.add_scalar("Metric/Val_F1", val_f1, epoch)
            self.tb_writer.add_scalar("LR", current_lr, epoch)

            # Log to CSV
            self.csv_logger.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "lr": current_lr
            })

            # Checkpoint management
            is_best = False
            monitor_val = val_f1 if self.es_monitor == "val_f1" else val_loss
            
            if self.es_mode == "max":
                if monitor_val > self.best_metric:
                    self.best_metric = monitor_val
                    is_best = True
            else:
                if monitor_val < self.best_metric:
                    self.best_metric = monitor_val
                    is_best = True

            self.save_checkpoint(epoch, is_best=is_best)

            # Early Stopping evaluation
            if is_best:
                self.es_counter = 0
            else:
                self.es_counter += 1
                if self.es_counter >= self.es_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs of training.")
                    break

        self.tb_writer.close()
        logger.info("Training process completed.")

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Saves current state and weights checkpoints.

        Args:
            epoch: Current training epoch index.
            is_best: True if validation score exceeds best recorded score.
        """
        from omegaconf import OmegaConf, DictConfig
        cfg_container = self.cfg
        if isinstance(self.cfg, DictConfig):
            cfg_container = OmegaConf.to_container(self.cfg, resolve=True)
            
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": cfg_container
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
            
        last_path = os.path.join(self.weights_dir, "last_model.pt")
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.weights_dir, "best_model.pt")
            torch.save(state, best_path)
            logger.debug(f"New best validation metric achieved. Saved to {best_path}")

    def resume_checkpoint(self, checkpoint_path: str) -> None:
        """Resumes model training from a stored .pt checkpoint.

        Args:
            checkpoint_path: Path to the .pt file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint["best_metric"]
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        logger.info(f"Successfully resumed training from epoch {self.start_epoch} (best score: {self.best_metric:.4f}).")
