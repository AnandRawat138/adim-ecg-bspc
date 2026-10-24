import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict

class BaseModel(nn.Module, ABC):
    """Abstract base class for all ECG classification models.

    Implements utility functions to profile model size, parameter counts, and structure.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_leads, signal_length).

        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        pass

    def get_parameter_count(self) -> Dict[str, int]:
        """Calculates total and trainable parameters of the model.

        Returns:
            Dict[str, int]: Dict containing 'total' and 'trainable' parameters count.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total_params,
            "trainable": trainable_params
        }

    def summarize(self) -> str:
        """Returns a string summarizing model layout and parameter stats."""
        stats = self.get_parameter_count()
        return (
            f"Architecture: {self.__class__.__name__}\n"
            f"---------------------------------\n"
            f"Total Parameters     : {stats['total']:,}\n"
            f"Trainable Parameters : {stats['trainable']:,}\n"
            f"Non-Trainable Params : {stats['total'] - stats['trainable']:,}\n"
            f"---------------------------------"
        )
