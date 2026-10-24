import torch
import torch.nn as nn
from typing import List

from models.base_model import BaseModel
from models.registry import ModelRegistry

class InceptionModule1D(nn.Module):
    """Inception Module with multi-scale 1D filters and bottleneck projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [9, 19, 39],
        bottleneck_channels: int = 32
    ) -> None:
        super().__init__()
        
        # Bottleneck projection
        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = in_channels

        # Parallel Convolutions with kernel sizes (9, 19, 39)
        self.convs = nn.ModuleList()
        for k_size in kernel_sizes:
            padding = (k_size - 1) // 2
            self.convs.append(
                nn.Conv1d(bottleneck_channels, out_channels, kernel_size=k_size, padding=padding, bias=False)
            )

        # MaxPooling shortcut branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Projection path
        projected = self.bottleneck(x)
        
        # Convolutions
        conv_outputs = [conv(projected) for conv in self.convs]
        
        # MaxPool path
        pooled = self.maxpool(x)
        pool_out = self.conv_pool(pooled)
        
        # Merge outputs (channels)
        combined = torch.cat(conv_outputs + [pool_out], dim=1)
        return self.relu(self.bn(combined))


class InceptionBlock1D(nn.Module):
    """Inception block encapsulating sequential inception modules with shortcut paths."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_modules: int = 3,
        kernel_sizes: List[int] = [9, 19, 39],
        bottleneck_channels: int = 32,
        use_residual: bool = True
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.modules_list = nn.ModuleList()
        
        current_in = in_channels
        for i in range(num_modules):
            # Inception module output size = 4 * out_channels
            module_out = out_channels
            self.modules_list.append(
                InceptionModule1D(
                    in_channels=current_in,
                    out_channels=module_out,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels
                )
            )
            current_in = module_out * 4
            
        # Downsample or project shortcut if dimension mismatch
        if self.use_residual:
            if in_channels != current_in:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, current_in, kernel_size=1, bias=False),
                    nn.BatchNorm1d(current_in)
                )
            else:
                self.shortcut = nn.Identity()
        else:
            self.shortcut = None

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x
        for module in self.modules_list:
            out = module(out)
            
        if self.use_residual and self.shortcut is not None:
            out = out + self.shortcut(identity)
            out = self.relu(out)
            
        return out


@ModelRegistry.register("inception_time")
class ECGInceptionTime(BaseModel):
    """InceptionTime sequence classification network adapted for 1D ECG signals."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        num_blocks: int = 6,
        filters: int = 32,
        kernel_sizes: List[int] = [9, 19, 39],
        bottleneck_channels: int = 32,
        use_residual: bool = True,
        dropout: float = 0.2,
        pooling: str = "avg",
        **kwargs
    ) -> None:
        super().__init__()
        
        # Build alternating inception blocks
        # Usually blocks are stacked, with residual links every 3 modules
        self.blocks = nn.ModuleList()
        in_c = input_channels
        
        # We divide num_blocks into pairs/groups of 3 modules
        num_modules_per_block = 3
        num_blocks_to_build = num_blocks // num_modules_per_block
        
        for _ in range(num_blocks_to_build):
            block = InceptionBlock1D(
                in_channels=in_c,
                out_channels=filters,
                num_modules=num_modules_per_block,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_residual=use_residual
            )
            self.blocks.append(block)
            in_c = filters * 4
            
        self.pool = nn.AdaptiveAvgPool1d(1) if pooling.lower() == "avg" else nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        out = self.pool(out).squeeze(-1)
        out = self.drop(out)
        return self.fc(out)
