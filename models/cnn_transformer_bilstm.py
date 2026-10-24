import torch
import torch.nn as nn
from typing import List

from models.base_model import BaseModel
from models.registry import ModelRegistry
from models.cnn_transformer import PositionalEncoding1D
from layers.recurrent import BiLSTMWrapper
from layers.attention import AttentionPooling1D

@ModelRegistry.register("cnn_transformer_bilstm")
class ECGCNNTransformerBILSTM(BaseModel):
    """ECG Model combining CNN convolutions, Transformer global self-attention, and BiLSTM temporal tracking."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        filters: List[int] = [64, 128],
        kernel_size: int = 5,
        stride: int = 2,
        dropout: float = 0.1,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        transformer_dropout: float = 0.1,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.2,
        pooling: str = "attention",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.pooling_type = pooling.lower()
        
        # CNN blocks
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
        
        # Project to Transformer Embedding
        self.projection = nn.Conv1d(in_c, embed_dim, kernel_size=1)
        self.pos_encoder = PositionalEncoding1D(embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # BiLSTM Layer wrapping
        self.bilstm = BiLSTMWrapper(
            in_channels=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout
        )
        
        lstm_out_dim = 2 * hidden_dim
        
        # Attention Pooling
        if self.pooling_type == "attention":
            self.pool = AttentionPooling1D(lstm_out_dim)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN Project
        feats = self.cnn_extractor(x)
        projected = self.projection(feats)
        
        # Transform: transpose to sequence layout
        seq = projected.transpose(1, 2)
        seq_encoded = self.pos_encoder(seq)
        trans_out = self.transformer_encoder(seq_encoded)
        
        # Back transpose to layout for LSTM
        trans_out_trans = trans_out.transpose(1, 2)
        
        # BiLSTM processing
        lstm_feats = self.bilstm(trans_out_trans)
        
        # Pool temporal axis
        if self.pooling_type == "attention":
            pooled = self.pool(lstm_feats)
        else:
            pooled = self.pool(lstm_feats).squeeze(-1)
            
        return self.classifier(pooled)
