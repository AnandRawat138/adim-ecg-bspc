import pytest
import torch
from layers.adim import AdaptiveDiagnosticInspectionModule

def test_adim_forward_and_shapes():
    """Verifies output shape and attention map extraction of ADIM."""
    batch_size = 4
    channels = 256
    time_steps = 100
    
    # Input tensor representing BiLSTM features: (B, D, T)
    x = torch.randn(batch_size, channels, time_steps)
    
    # Instantiate ADIM
    adim = AdaptiveDiagnosticInspectionModule(
        input_dim=channels,
        reduction_ratio=8,
        attention_dim=64
    )
    
    # Run forward pass
    out = adim(x)
    
    # Check shape
    assert out.shape == (batch_size, channels, time_steps)
    assert not torch.isnan(out).any(), "NaN values found in ADIM output"
    
    # Check attention map extraction
    attn_map = adim.get_last_attention_map()
    assert attn_map is not None
    assert attn_map.shape == (batch_size, time_steps)
    # Check softmax sum property
    assert torch.allclose(attn_map.sum(dim=-1), torch.ones(batch_size))


def test_adim_ablation_switches():
    """Verifies that ADIM switches disable respective modules successfully without errors."""
    batch_size = 2
    channels = 128
    time_steps = 50
    x = torch.randn(batch_size, channels, time_steps)
    
    # 1. Test w/o LGIG
    adim_no_lgig = AdaptiveDiagnosticInspectionModule(input_dim=channels, disable_lgig=True)
    out1 = adim_no_lgig(x)
    assert out1.shape == x.shape
    
    # 2. Test w/o AIN
    adim_no_ain = AdaptiveDiagnosticInspectionModule(input_dim=channels, disable_ain=True)
    out2 = adim_no_ain(x)
    assert out2.shape == x.shape
    
    # 3. Test w/o Certainty
    adim_no_cert = AdaptiveDiagnosticInspectionModule(input_dim=channels, disable_certainty=True)
    out3 = adim_no_cert(x)
    assert out3.shape == x.shape
    
    # 4. Test w/o Residual
    adim_no_res = AdaptiveDiagnosticInspectionModule(input_dim=channels, disable_residual=True)
    out4 = adim_no_res(x)
    assert out4.shape == x.shape
