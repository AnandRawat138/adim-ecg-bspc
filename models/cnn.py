import torch
import torch.nn as nn
from typing import List, Optional

from models.base_model import BaseModel
from models.registry import ModelRegistry
from layers.convolutions import ResidualBlock1D
from layers.attention import SqueezeExcitation1D, CBAM1D, AttentionPooling1D

@ModelRegistry.register("cnn")
class ECGCNN(BaseModel):
    """Convolutional Neural Network supporting Residual connections, Depthwise Separable, and SE/CBAM attention."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        filters: List[int] = [64, 128, 256, 512],
        kernel_size: int = 5,
        stride: int = 2,
        dropout: float = 0.2,
        residual_connections: bool = True,
        depthwise_separable: bool = False,
        squeeze_excitation: bool = True,
        cbam: bool = False,
        pooling: str = "attention",
        fc_dim: int = 128,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.pooling_type = pooling.lower()
        self.residual_connections = residual_connections
        
        layers = []
        in_c = input_channels
        
        for i, out_c in enumerate(filters):
            # Setup attention blocks for residual blocks if enabled
            se = SqueezeExcitation1D(out_c) if squeeze_excitation else None
            cbam_attn = CBAM1D(out_c) if cbam else None
            
            block_stride = stride if i > 0 else 1
            
            if residual_connections:
                layers.append(
                    ResidualBlock1D(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        dropout=dropout,
                        depthwise_separable=depthwise_separable,
                        se_block=se,
                        cbam_block=cbam_attn
                    )
                )
            else:
                # Sequential basic block
                conv_layer = nn.Conv1d(
                    in_c, out_c, kernel_size, stride=block_stride, padding=(kernel_size - 1) // 2
                ) if not depthwise_separable else nn.Sequential(
                    nn.Conv1d(in_c, in_c, kernel_size, stride=block_stride, padding=(kernel_size - 1) // 2, groups=in_c),
                    nn.Conv1d(in_c, out_c, 1)
                )
                
                layers.extend([
                    conv_layer,
                    nn.BatchNorm1d(out_c),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                
                if se is not None:
                    layers.append(se)
                if cbam_attn is not None:
                    layers.append(cbam_attn)
                    
            in_c = out_c
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Pooling method
        if self.pooling_type == "attention":
            self.pool = AttentionPooling1D(in_c)
        elif self.pooling_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)
            
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(in_c, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        feats = self.feature_extractor(x)
        
        # Pool temporal axis
        if self.pooling_type == "attention":
            pooled = self.pool(feats)
        else:
            pooled = self.pool(feats).squeeze(-1)
            
        # Classify
        return self.fc(pooled)
