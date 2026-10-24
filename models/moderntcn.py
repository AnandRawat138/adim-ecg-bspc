import torch
import torch.nn as nn
from typing import List

from models.base_model import BaseModel
from models.registry import ModelRegistry

class ModernTCNBlock1D(nn.Module):
    """1D ModernTCN Block separating channel mixing and spatial mixing with large kernels."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 17,
        expansion_factor: int = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # Spatial mixing: Depthwise Conv with large kernel
        self.spatial_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=dim, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        
        # Channel mixing: Inverted Bottleneck
        self.channel_mixing = nn.Sequential(
            nn.Conv1d(dim, dim * expansion_factor, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * expansion_factor, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial representation step with skip connection
        x_spatial = x + self.spatial_conv(x)
        x_norm = self.norm(x_spatial)
        
        # Channel mixing step with skip connection
        out = x_norm + self.channel_mixing(x_norm)
        return out


@ModelRegistry.register("moderntcn")
class ModernTCN1D(BaseModel):
    """ModernTCN sequence model for ECG classification."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        depths: List[int] = [2, 2, 2, 2],
        dims: List[int] = [64, 128, 256, 512],
        kernel_size: int = 17,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        pooling: str = "avg",
        **kwargs
    ) -> None:
        super().__init__()
        
        # Initial projection stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, dims[0], kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(num_groups=1, num_channels=dims[0]),
            nn.GELU()
        )
        
        # Downsample and feature blocks stack
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        in_dim = dims[0]
        for i in range(len(depths)):
            # Downsampling layer between stages (except first stage)
            if i > 0:
                downsample = nn.Sequential(
                    nn.GroupNorm(num_groups=1, num_channels=in_dim),
                    nn.Conv1d(in_dim, dims[i], kernel_size=2, stride=2)
                )
                in_dim = dims[i]
            else:
                downsample = nn.Identity()
                
            self.downsample_layers.append(downsample)
            
            # ModernTCN blocks for this stage
            blocks = nn.Sequential(*[
                ModernTCNBlock1D(
                    dim=in_dim,
                    kernel_size=kernel_size,
                    expansion_factor=expansion_factor,
                    dropout=dropout
                ) for _ in range(depths[i])
            ])
            self.stages.append(blocks)
            
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_dim)
        self.pool = nn.AdaptiveAvgPool1d(1) if pooling.lower() == "avg" else nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        
        for i in range(len(self.stages)):
            out = self.downsample_layers[i](out)
            out = self.stages[i](out)
            
        out = self.norm(out)
        out = self.pool(out).squeeze(-1)
        return self.classifier(out)
