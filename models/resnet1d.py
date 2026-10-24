import torch
import torch.nn as nn
from typing import List

from models.base_model import BaseModel
from models.registry import ModelRegistry

class ResNetBasicBlock1D(nn.Module):
    """Basic 1D Residual Block with standard convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.relu(out)
        return out


@ModelRegistry.register("resnet1d")
class ResNet1D(BaseModel):
    """1D implementation of the standard ResNet architecture."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        blocks: List[int] = [2, 2, 2, 2],
        filters: List[int] = [64, 128, 256, 512],
        kernel_size: int = 7,
        stride: int = 2,
        dropout: float = 0.1,
        pooling: str = "avg",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.in_channels = filters[0]
        
        # Stem Layer
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, self.in_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual stages
        stages = []
        for i, num_blocks in enumerate(blocks):
            stage_stride = stride if i > 0 else 1
            stage_layers = []
            
            # First block in stage handles channel projection and downsampling
            stage_layers.append(
                ResNetBasicBlock1D(
                    in_channels=self.in_channels,
                    out_channels=filters[i],
                    kernel_size=kernel_size,
                    stride=stage_stride,
                    dropout=dropout
                )
            )
            self.in_channels = filters[i]
            
            # Subsequent blocks keep channels/sizes same
            for _ in range(1, num_blocks):
                stage_layers.append(
                    ResNetBasicBlock1D(
                        in_channels=self.in_channels,
                        out_channels=self.in_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dropout=dropout
                    )
                )
            stages.append(nn.Sequential(*stage_layers))
            
        self.residual_stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1) if pooling.lower() == "avg" else nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(filters[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.residual_stages(out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)
