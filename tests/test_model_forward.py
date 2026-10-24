import pytest
import torch
from models.registry import ModelRegistry

# Direct imports to force registration in tests
import models

@pytest.mark.parametrize("model_name", [
    "cnn",
    "cnn_transformer",
    "cnn_bilstm",
    "cnn_transformer_bilstm",
    "attention_hybrid",
    "resnet1d",
    "inception_time",
    "xresnet1d",
    "convnext1d",
    "moderntcn",
    "inception_time_hybrid"
])
def test_all_models_forward_pass(model_name):
    """Verifies that all models in the model zoo instantiate and run forward passes."""
    batch_size = 4
    num_leads = 12
    signal_length = 1000
    num_classes = 5
    
    # Create random input tensor
    x = torch.randn(batch_size, num_leads, signal_length)
    
    # Define configurations for instantiation (fallback settings)
    kwargs = {
        "input_channels": num_leads,
        "num_classes": num_classes,
    }
    
    # Instantiate from registry
    model = ModelRegistry.create(model_name, **kwargs)
    
    # Run forward pass
    out = model(x)
    
    # Verify shape consistency
    assert out.shape == (batch_size, num_classes)
    assert not torch.isnan(out).any(), f"NaN values detected in output of {model_name}"
    
    # Verify parameter summary works
    summary = model.summarize()
    assert "Architecture" in summary
    assert "Total Parameters" in summary
