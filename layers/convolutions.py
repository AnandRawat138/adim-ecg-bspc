import torch
import torch.nn as nn
from typing import Optional

class DepthwiseSeparableConv1D(nn.Module):
    """1D Depthwise Separable Convolution layer.

    Factorizes a standard convolution into a depthwise convolution
    (filtering channels individually) and a pointwise convolution
    (combining channels linearly). Reduces parameter size and compute.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualBlock1D(nn.Module):
    """1D Residual block with optional Squeeze-and-Excitation / CBAM and Separable Convs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
        depthwise_separable: bool = False,
        se_block: Optional[nn.Module] = None,
        cbam_block: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        # Conv layer selection
        if depthwise_separable:
            self.conv1 = DepthwiseSeparableConv1D(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            self.conv2 = DepthwiseSeparableConv1D(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
            
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        
        self.se = se_block
        self.cbam = cbam_block
        
        # Shortcut link
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
        
        # Apply spatial/channel attention
        if self.se is not None:
            out = self.se(out)
        if self.cbam is not None:
            out = self.cbam(out)
            
        out += identity
        out = self.relu(out)
        return out
