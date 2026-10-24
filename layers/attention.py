import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation block for 1D signals.

    Re-calibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.shape
        # Squeeze
        y = self.avg_pool(x).view(batch, channels)
        # Excitation
        y = self.fc(y).view(batch, channels, 1)
        return x * y


class ChannelAttention1D(nn.Module):
    """Channel attention component of CBAM for 1D signals."""

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.shape
        avg_out = self.shared_mlp(self.avg_pool(x).view(batch, channels))
        max_out = self.shared_mlp(self.max_pool(x).view(batch, channels))
        
        out = self.sigmoid(avg_out + max_out).view(batch, channels, 1)
        return x * out


class SpatialAttention1D(nn.Module):
    """Spatial attention component of CBAM for 1D signals."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in [3, 7], "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Max pool and Avg pool across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(combined)
        return x * self.sigmoid(out)


class CBAM1D(nn.Module):
    """Convolutional Block Attention Module for 1D signals.

    Sequentially applies channel-wise and spatial-wise attention.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention1D(channels, reduction_ratio)
        self.spatial_attn = SpatialAttention1D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attn(x)
        out = self.spatial_attn(out)
        return out


class AttentionPooling1D(nn.Module):
    """Attention Pooling layer to collapse the time dimension of an ECG sequence.

    Uses a learned query context vector to perform a weighted average over time steps.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of AttentionPooling.

        Args:
            x: Input of shape (batch_size, channels, time_steps).

        Returns:
            torch.Tensor: Aggregated representation of shape (batch_size, channels).
        """
        # Shape: (batch_size, time_steps, channels)
        x_trans = x.transpose(1, 2)
        
        # Compute attention scores. Shape: (batch_size, time_steps, 1)
        scores = self.query_proj(x_trans)
        
        # Softmax over time_steps
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum. Shape: (batch_size, channels)
        weighted_sum = torch.sum(x_trans * weights, dim=1)
        return weighted_sum
