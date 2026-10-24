import torch
import torch.nn as nn
import math
from typing import List, Optional

from models.base_model import BaseModel
from models.registry import ModelRegistry
from layers.attention import AttentionPooling1D

class PositionalEncoding1D(nn.Module):
    """Standard 1D Positional Encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 2000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd embedding sizes safely
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


@ModelRegistry.register("cnn_transformer")
class ECGCNNTransformer(BaseModel):
    """ECG Model combining CNN local feature extraction with global Transformer mapping."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        filters: List[int] = [64, 128, 256],
        kernel_size: int = 5,
        stride: int = 2,
        dropout: float = 0.1,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        transformer_dropout: float = 0.1,
        pooling: str = "attention",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.pooling_type = pooling.lower()
        
        # 1D Convolutional projection blocks
        layers = []
        in_c = input_channels
        for out_c in filters:
            layers.extend([
                nn.Conv1d(in_c, out_c, kernel_size, stride=stride, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_c = out_c
            
        self.cnn_extractor = nn.Sequential(*layers)
        
        # Project CNN output to transformer embedding dimension
        self.projection = nn.Conv1d(in_c, embed_dim, kernel_size=1)
        self.pos_encoder = PositionalEncoding1D(embed_dim)
        
        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling block
        if self.pooling_type == "attention":
            self.pool = AttentionPooling1D(embed_dim)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN forward: outputs (batch, filters[-1], time)
        feats = self.cnn_extractor(x)
        
        # Projection: outputs (batch, embed_dim, time)
        projected = self.projection(feats)
        
        # Shape change to (batch, time, embed_dim) for Transformer
        seq = projected.transpose(1, 2)
        seq_encoded = self.pos_encoder(seq)
        
        # Transformer encoding
        trans_out = self.transformer_encoder(seq_encoded)
        
        # Transpose back to (batch, embed_dim, time) for pooling
        trans_out_trans = trans_out.transpose(1, 2)
        
        # Pool temporal axis
        if self.pooling_type == "attention":
            pooled = self.pool(trans_out_trans)
        else:
            pooled = self.pool(trans_out_trans).squeeze(-1)
            
        return self.classifier(pooled)
