import torch
import torch.nn as nn
from typing import List

from models.base_model import BaseModel
from models.registry import ModelRegistry

class XResNetBlock1D(nn.Module):
    """1D XResNet Basic Block implementing ResNet-B modifications (stride-shift)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        # ResNet-B modification: downsampling is applied on the second conv
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # ResNet-D modification: shortcut pathway downsamples via pooling
        if stride != 1 or in_channels != out_channels:
            shortcut_layers = []
            if stride != 1:
                shortcut_layers.append(nn.AvgPool1d(kernel_size=2*stride-1, stride=stride, padding=stride-1, count_include_pad=False))
            shortcut_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            ])
            self.shortcut = nn.Sequential(*shortcut_layers)
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


@ModelRegistry.register("xresnet1d")
class XResNet1D(BaseModel):
    """XResNet1D implementing deep bag-of-tricks architectural modifications (ResNet-B and D)."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        blocks: List[int] = [3, 4, 6, 3],
        filters: List[int] = [32, 64, 128, 256, 512, 1024],
        kernel_size: int = 5,
        stride: int = 2,
        dropout: float = 0.15,
        pooling: str = "avg",
        **kwargs
    ) -> None:
        super().__init__()
        
        # XResNet Stem modification: replace 7x7 conv with three sequential 3x3 convs
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, filters[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters[0], filters[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(filters[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        in_c = filters[1]
        
        stages = []
        for i, num_blocks in enumerate(blocks):
            stage_stride = stride
            stage_layers = []
            
            stage_layers.append(
                XResNetBlock1D(
                    in_channels=in_c,
                    out_channels=filters[i + 2],
                    kernel_size=kernel_size,
                    stride=stage_stride,
                    dropout=dropout
                )
            )
            in_c = filters[i + 2]
            
            for _ in range(1, num_blocks):
                stage_layers.append(
                    XResNetBlock1D(
                        in_channels=in_c,
                        out_channels=in_c,
                        kernel_size=kernel_size,
                        stride=1,
                        dropout=dropout
                    )
                )
            stages.append(nn.Sequential(*stage_layers))
            
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1) if pooling.lower() == "avg" else nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(in_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stages(out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)
