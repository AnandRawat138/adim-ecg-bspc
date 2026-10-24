import torch
import torch.nn as nn
from typing import List

from models.base_model import BaseModel
from models.registry import ModelRegistry
from models.inception_time import InceptionBlock1D
from models.cnn_transformer import PositionalEncoding1D
from layers.recurrent import BiLSTMWrapper
from layers.attention import AttentionPooling1D
from layers.adim import AdaptiveDiagnosticInspectionModule
from layers.refinement import build_refinement

@ModelRegistry.register("inception_time_hybrid")
class ECGInceptionTimeHybrid(BaseModel):
    """ECG InceptionTime Hybrid Model.
    
    Dynamically configures stacking:
    InceptionTime Backbone -> Transformer -> BiLSTM -> ADIM -> Pooling -> Classifier
    Supports optional sub-ADIM ablation study configurations.
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        num_blocks: int = 6,
        filters: int = 32,
        kernel_sizes: List[int] = [9, 19, 39],
        bottleneck_channels: int = 32,
        use_residual: bool = True,
        use_transformer: bool = False,
        use_bilstm: bool = False,
        use_adim: bool = False,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 512,
        transformer_dropout: float = 0.1,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.2,
        adim_reduction_ratio: int = 8,
        adim_attention_dim: int = 64,
        adim_dropout: float = 0.1,
        disable_lgig: bool = False,
        disable_ain: bool = False,
        disable_certainty: bool = False,
        disable_residual: bool = False,
        refinement: str = "none",
        adim_formulation: str = "v1",
        pooling: str = "avg",
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.use_transformer = use_transformer
        self.use_bilstm = use_bilstm
        self.use_adim = use_adim
        self.refinement_type = refinement.lower()
        assert not (use_adim and self.refinement_type != "none"), "Use either ADIM or a baseline refinement, not both"
        self.pooling_type = pooling.lower()
        
        # 1. InceptionTime Backbone
        self.blocks = nn.ModuleList()
        in_c = input_channels
        
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
            
        curr_dim = in_c
        
        # 2. Transformer Block
        if self.use_transformer:
            self.trans_proj = nn.Identity()
            self.pos_encoder = PositionalEncoding1D(curr_dim)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=curr_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            
        # 3. BiLSTM Block
        if self.use_bilstm:
            self.bilstm = BiLSTMWrapper(
                in_channels=curr_dim,
                hidden_dim=lstm_hidden_dim,
                num_layers=lstm_layers,
                dropout=lstm_dropout
            )
            curr_dim = 2 * lstm_hidden_dim
            
        # 4. ADIM Block
        if self.use_adim:
            self.adim = AdaptiveDiagnosticInspectionModule(
                input_dim=curr_dim,
                reduction_ratio=adim_reduction_ratio,
                attention_dim=adim_attention_dim,
                dropout=adim_dropout,
                disable_lgig=disable_lgig,
                disable_ain=disable_ain,
                disable_certainty=disable_certainty,
                disable_residual=disable_residual,
                formulation=adim_formulation
            )
            
        # 4b. Baseline refinement (SE / CBAM / temporal attention) at the ADIM position
        if self.refinement_type != "none":
            self.refine = build_refinement(
                self.refinement_type, curr_dim,
                reduction_ratio=adim_reduction_ratio,
                attention_dim=adim_attention_dim,
                dropout=adim_dropout
            )

        # 5. Pooling Block
        if self.pooling_type == "attention":
            self.pool = AttentionPooling1D(curr_dim)
        elif self.pooling_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            
        # 6. Classifier
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(curr_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. InceptionTime Backbone features
        out = x
        for block in self.blocks:
            out = block(out)  # (B, C_inception, T_out)
            
        # 2. Transformer global attention
        if self.use_transformer:
            out = self.trans_proj(out)
            seq = out.transpose(1, 2)  # (B, T_out, C_proj)
            seq_encoded = self.pos_encoder(seq)
            trans_out = self.transformer_encoder(seq_encoded)
            out = trans_out.transpose(1, 2)  # (B, C_proj, T_out)
            
        # 3. BiLSTM sequential dependencies
        if self.use_bilstm:
            out = self.bilstm(out)  # (B, 2*lstm_hidden_dim, T_out)
            
        # 4. ADIM sequence refinement
        if self.use_adim:
            out = self.adim(out)  # (B, D_curr, T_out)
        elif self.refinement_type != "none":
            out = self.refine(out)
            
        # 5. Pooling
        if self.pooling_type == "attention":
            pooled = self.pool(out)  # (B, D_curr)
        else:
            pooled = self.pool(out).squeeze(-1)  # (B, D_curr)
            
        # 6. Classification
        pooled = self.drop(pooled)
        return self.classifier(pooled)
