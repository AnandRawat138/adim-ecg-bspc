import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import SqueezeExcitation1D, CBAM1D


class TemporalAttentionRefinement(nn.Module):
    """Conventional additive temporal attention used as a sequence-refinement baseline.

    a = softmax_t(w^T tanh(W h_t)),   h'_t = T * a_t * h_t
    The factor T keeps the mean temporal weight equal to one, so the block reweights time steps
    without changing the overall feature scale.

    Shape:
        Input: (B, D, T)
        Output: (B, D, T)
    """

    def __init__(self, input_dim: int, attention_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1)
        )
        self.last_attention_map = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        scores = self.scorer(x.transpose(1, 2)).squeeze(-1)  # (B, T)
        a = F.softmax(scores, dim=-1)
        self.last_attention_map = a
        return x * (T * a).unsqueeze(1)


def build_refinement(name: str, input_dim: int, reduction_ratio: int = 8, attention_dim: int = 64,
                     dropout: float = 0.1) -> nn.Module:
    """Factory for non-ADIM refinement baselines inserted at the ADIM position."""
    name = name.lower()
    if name == "se":
        return SqueezeExcitation1D(input_dim, reduction_ratio=reduction_ratio)
    if name == "cbam":
        return CBAM1D(input_dim, reduction_ratio=reduction_ratio)
    if name == "tattn":
        return TemporalAttentionRefinement(input_dim, attention_dim=attention_dim, dropout=dropout)
    raise ValueError(f"Unknown refinement '{name}'")
