import torch
import torch.nn as nn
from typing import List

from models.base_model import BaseModel
from models.registry import ModelRegistry
from layers.recurrent import BiLSTMWrapper
from layers.attention import AttentionPooling1D

@ModelRegistry.register("cnn_bilstm")
class ECGCNNBILSTM(BaseModel):
    """ECG Model combining CNN local convolutions with BiLSTM temporal mapping."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        filters: List[int] = [64, 128, 256],
        kernel_size: int = 5,
        stride: int = 2,
        dropout: float = 0.1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        lstm_dropout: float = 0.2,
        pooling: str = "attention",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.pooling_type = pooling.lower()
        
        # CNN Projection Blocks
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
        
        # BiLSTM Layer
        self.bilstm = BiLSTMWrapper(
            in_channels=in_c,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout
        )
        
        # BiLSTM output feature size = 2 * hidden_dim
        lstm_out_dim = 2 * hidden_dim
        
        # Pooling Block
        if self.pooling_type == "attention":
            self.pool = AttentionPooling1D(lstm_out_dim)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN features
        feats = self.cnn_extractor(x)
        
        # BiLSTM forward
        lstm_feats = self.bilstm(feats)
        
        # Pool temporal axis
        if self.pooling_type == "attention":
            pooled = self.pool(lstm_feats)
        else:
            pooled = self.pool(lstm_feats).squeeze(-1)
            
        return self.classifier(pooled)
