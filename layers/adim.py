import torch
import torch.nn as nn
import math

class AdaptiveDiagnosticInspectionModule(nn.Module):
    """Adaptive Diagnostic Inspection Module (ADIM).
    
    Operates on temporal sequence representation after BiLSTM layers.
    Performs Local-Global Inspection Gating, multi-descriptor Adaptive Inspection,
    Attention Certainty Estimation, and Certainty-Calibrated Residual Refinement.
    Supports ablation testing switches to disable components.
    
    Shape:
        Input: (B, D, T) - where B is batch size, D is feature dimension, T is time steps
        Output: (B, D, T) - same shape as input
    """
    
    def __init__(
        self,
        input_dim: int,
        reduction_ratio: int = 8,
        attention_dim: int = 64,
        dropout: float = 0.1,
        disable_lgig: bool = False,
        disable_ain: bool = False,
        disable_certainty: bool = False,
        disable_residual: bool = False,
        formulation: str = "v1"
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.reduction_ratio = reduction_ratio
        self.attn_dim = attention_dim
        
        self.disable_lgig = disable_lgig
        self.disable_ain = disable_ain
        self.disable_certainty = disable_certainty
        self.disable_residual = disable_residual
        # "v1": original formulation, LN(H_c * (1 + gamma*c*a)) -> projection. A positive per-step scale is
        #       removed by the LayerNorm, so the inspection path cannot change the output.
        # "v2": revised formulation, the temporal weight w_t = 1 + gamma*(1+c)*(T*a_t - 1) multiplies the
        #       projected, normalised branch: H_out = H + w * (W_p LN(H_c) + b_p). w = 1 for a uniform map.
        assert formulation in ("v1", "v2")
        self.formulation = formulation
        
        # 1. Local-Global Inspection Gate (LGIG)
        bottleneck_dim = max(input_dim // reduction_ratio, 8)
        self.lgig = nn.Sequential(
            nn.Linear(input_dim * 2, bottleneck_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(bottleneck_dim, input_dim),
            nn.Sigmoid()
        )
        
        # 2. Adaptive Inspection Network (AIN) Descriptors
        # Local Morphology: 1D Conv with kernel size 3
        self.morph_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=attention_dim,
            kernel_size=3,
            padding=1
        )
        
        # Global Context Projection
        self.context_proj = nn.Linear(input_dim, attention_dim)
        
        # Temporal Consistency (Difference): 1D Conv with kernel size 3
        self.temp_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=attention_dim,
            kernel_size=3,
            padding=1
        )
        
        # Inspection Map scoring projection
        total_descriptor_dim = attention_dim * 3
        self.inspection_scorer = nn.Sequential(
            nn.Linear(total_descriptor_dim, attention_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1)
        )
        
        # Simple fallback projection (if AIN is disabled)
        self.fallback_scorer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1)
        )
        
        # 3. Residual Refinement components
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc_proj = nn.Linear(input_dim, input_dim)
        
        # Placeholder for extracting attention maps during evaluation/interpretability visualization
        self.last_attention_map = None

    def get_last_attention_map(self) -> torch.Tensor:
        """Returns the detached, numpy-converted last computed attention map."""
        if self.last_attention_map is not None:
            return self.last_attention_map.detach().cpu()
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape x: (B, D, T)
        B, D, T = x.shape
        
        # Transpose to channel-last: (B, T, D)
        h = x.transpose(1, 2)
        
        # Extract mean and std dev stats
        mu = h.mean(dim=1)  # (B, D)
        sigma = torch.sqrt(h.var(dim=1, unbiased=False) + 1e-9)  # (B, D)
        
        # --- Step 1: Local-Global Inspection Gate (LGIG) ---
        if self.disable_lgig:
            h_c = h
        else:
            stats = torch.cat([mu, sigma], dim=-1)  # (B, 2D)
            g_c = self.lgig(stats)  # (B, D)
            h_c = h * g_c.unsqueeze(1)
            
        # --- Step 2: Adaptive Inspection Network (AIN) ---
        if self.disable_ain:
            # Fallback simple scoring: project calibrated sequence directly
            scores = self.fallback_scorer(h_c).squeeze(-1)  # (B, T)
        else:
            h_c_t = h_c.transpose(1, 2)
            
            # Local Morphology Descriptor: (B, D_m, T) -> transpose to (B, T, D_m)
            m = self.morph_conv(h_c_t).transpose(1, 2)
            
            # Global Context Descriptor: (B, D) -> project to (B, D_c) -> (B, 1, D_c) -> repeat over time axis to (B, T, D_c)
            c_g = self.context_proj(mu).unsqueeze(1).repeat(1, T, 1)
            
            # Temporal Consistency (Difference) Descriptor:
            # First-order temporal difference: (B, T, D)
            diff = torch.zeros_like(h_c_t)
            diff[:, :, 1:] = h_c_t[:, :, 1:] - h_c_t[:, :, :-1]
            t_c = self.temp_conv(diff).transpose(1, 2)  # (B, T, D_t)
            
            # Concat descriptors: (B, T, D_m + D_c + D_t)
            v = torch.cat([m, c_g, t_c], dim=-1)
            
            # Project to raw inspection scores: (B, T, 1) -> squeeze to (B, T)
            scores = self.inspection_scorer(v).squeeze(-1)
            
        # Compute inspection map a: (B, T)
        a = torch.softmax(scores, dim=-1)
        self.last_attention_map = a
        
        # --- Step 3: Inspection Certainty Estimation (ICE) ---
        if self.disable_certainty:
            # v1 ablation fixes c = 1; v2 ablation removes the concentration modulation (c = 0)
            fill = 1.0 if self.formulation == "v1" else 0.0
            s_certain = torch.full((B, 1), fill, device=x.device)
        else:
            entropy = -torch.sum(a * torch.log(a + 1e-9), dim=-1)  # (B,)
            log_t = math.log(T)
            s_certain = 1.0 - (entropy / log_t)
            s_certain = s_certain.unsqueeze(1)  # (B, 1)
            
        # --- Step 4: Certainty-Calibrated Residual Refinement ---
        if self.formulation == "v1":
            # Refined sequence: (B, T, D)
            refined = h_c * (1.0 + self.gamma * s_certain.unsqueeze(2) * a.unsqueeze(2))
            normed = self.layer_norm(refined)
            proj_out = self.fc_proj(normed)  # (B, T, D)
        else:
            w = 1.0 + self.gamma * (1.0 + s_certain) * (T * a - 1.0)  # (B, T)
            self.last_temporal_weight = w
            proj_out = w.unsqueeze(2) * self.fc_proj(self.layer_norm(h_c))
        
        if self.disable_residual:
            out = proj_out
        else:
            out = h + proj_out
            
        return out.transpose(1, 2)
