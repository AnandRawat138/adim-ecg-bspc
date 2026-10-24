import torch
import torch.nn as nn

class BiLSTMWrapper(nn.Module):
    """Bidirectional LSTM wrapper designed to work with 1D convolutional tensor layouts.

    Accepts shape: (batch_size, channels, time_steps)
    Outputs shape: (batch_size, 2 * hidden_dim, time_steps)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, time_steps)

        Returns:
            torch.Tensor: LSTM output of shape (batch_size, 2 * hidden_dim, time_steps)
        """
        # LSTM expects (batch, seq_len, input_size)
        x_trans = x.transpose(1, 2)
        
        # Out shape: (batch, seq_len, 2 * hidden_dim)
        out, _ = self.lstm(x_trans)
        
        # Transpose back to match Conv1D output shape (batch, channels, seq_len)
        return out.transpose(1, 2)
