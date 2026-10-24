import torch
import torch.nn as nn
from typing import List

from models.base_model import BaseModel
from models.registry import ModelRegistry

class ConvNeXtBlock1D(nn.Module):
    """1D ConvNeXt Block mapping channel-mixing and depthwise convolutions."""

    def __init__(self, dim: int, drop_path: float = 0.0, kernel_size: int = 7) -> None:
        super().__init__()
        # Depthwise conv with large kernel size
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=dim)
        
        # LayerNorm (channels last layout is standard in PyTorch ConvNeXt, but we can implement LayerNorm1D)
        self.layernorm = nn.GroupNorm(num_groups=1, num_channels=dim) # GroupNorm with G=1 is identical to LayerNorm
        
        # Pointwise channel mixing (inverted bottleneck)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)
        
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.dwconv(x)
        out = self.layernorm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        
        out = identity + self.drop_path(out)
        return out


@ModelRegistry.register("convnext1d")
class ConvNeXt1D(BaseModel):
    """1D ConvNeXt classification architecture for ECG sequences."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.1,
        kernel_size: int = 7,
        pooling: str = "avg",
        **kwargs
    ) -> None:
        super().__init__()
        
        # Stem patchify: 4x downsampling
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=1, num_channels=dims[0])
        )
        
        # Downsample stages (applied between blocks)
        self.downsample_layers = nn.ModuleList()
        # First stem downsample is identity since stem already projects to dim[0]
        self.downsample_layers.append(nn.Identity())
        
        for i in range(3):
            downsample = nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)
            
        # ConvNeXt Blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        block_idx = 0
        for i in range(4):
            stage_blocks = []
            for _ in range(depths[i]):
                stage_blocks.append(
                    ConvNeXtBlock1D(dim=dims[i], drop_path=dp_rates[block_idx], kernel_size=kernel_size)
                )
                block_idx += 1
            self.stages.append(nn.Sequential(*stage_blocks))
            
        self.layernorm = nn.GroupNorm(num_groups=1, num_channels=dims[-1])
        self.pool = nn.AdaptiveAvgPool1d(1) if pooling.lower() == "avg" else nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        
        for stage_idx in range(4):
            out = self.downsample_layers[stage_idx](out)
            out = self.stages[stage_idx](out)
            
        out = self.layernorm(out)
        out = self.pool(out).squeeze(-1)
        return self.classifier(out)
