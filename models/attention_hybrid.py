import torch
import torch.nn as nn
from typing import List

from models.base_model import BaseModel
from models.registry import ModelRegistry
from models.cnn_transformer import PositionalEncoding1D
from layers.attention import AttentionPooling1D

@ModelRegistry.register("attention_hybrid")
class ECGAttentionHybrid(BaseModel):
    """Parallel multi-scale CNN feature extractors coupled with Hierarchical Attention mapping."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        filters: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.2,
        attention_dim: int = 128,
        num_heads: int = 8,
        pooling: str = "attention",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.pooling_type = pooling.lower()
        
        # Parallel Multi-Scale Convolutions
        self.branches = nn.ModuleList()
        for k_size in kernel_sizes:
            padding = (k_size - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(input_channels, filters[0], kernel_size=k_size, padding=padding),
                nn.BatchNorm1d(filters[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.branches.append(branch)
            
        # Deeper shared sequential block
        shared_layers = []
        in_c = filters[0] * len(kernel_sizes)
        for out_c in filters[1:]:
            shared_layers.extend([
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_c = out_c
            
        self.shared_convs = nn.Sequential(*shared_layers)
        
        # Self-Attention Layer projection
        self.attention_proj = nn.Conv1d(in_c, attention_dim, kernel_size=1)
        self.pos_encoder = PositionalEncoding1D(attention_dim)
        
        # Standard Multihead Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(attention_dim)
        
        # Temporal Pooling
        if self.pooling_type == "attention":
            self.pool = AttentionPooling1D(attention_dim)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            
        self.classifier = nn.Linear(attention_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale feature branches extraction
        branch_feats = [branch(x) for branch in self.branches]
        
        # Concatenate features along the channel axis
        concat_feats = torch.cat(branch_feats, dim=1)
        
        # Deep shared convolutions
        shared_feats = self.shared_convs(concat_feats)
        
        # Projection to attention shape
        attn_in = self.attention_proj(shared_feats)
        
        # Prepare for Multi-head Attention
        seq = attn_in.transpose(1, 2)
        seq_encoded = self.pos_encoder(seq)
        
        # Self-Attention mapping with skip connection and Norm
        attn_out, _ = self.self_attn(seq_encoded, seq_encoded, seq_encoded)
        attn_norm = self.layer_norm(seq_encoded + attn_out)
        
        # Transpose back for pooling
        feats_out = attn_norm.transpose(1, 2)
        
        # Pool temporal dimensions
        if self.pooling_type == "attention":
            pooled = self.pool(feats_out)
        else:
            pooled = self.pool(feats_out).squeeze(-1)
            
        return self.classifier(pooled)
