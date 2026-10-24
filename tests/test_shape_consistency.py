import pytest
import torch
from layers.convolutions import DepthwiseSeparableConv1D, ResidualBlock1D
from layers.attention import SqueezeExcitation1D, CBAM1D, AttentionPooling1D
from layers.recurrent import BiLSTMWrapper

def test_depthwise_separable_shape():
    x = torch.randn(4, 16, 100)
    layer = DepthwiseSeparableConv1D(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
    out = layer(x)
    assert out.shape == (4, 32, 50)


def test_residual_block_shape():
    x = torch.randn(4, 16, 100)
    # Testing stride 1
    block1 = ResidualBlock1D(in_channels=16, out_channels=16, stride=1)
    assert block1(x).shape == (4, 16, 100)
    
    # Testing stride 2 with channel expansion
    block2 = ResidualBlock1D(in_channels=16, out_channels=32, stride=2)
    assert block2(x).shape == (4, 32, 50)


def test_squeeze_excitation_shape():
    x = torch.randn(4, 16, 100)
    se = SqueezeExcitation1D(channels=16, reduction_ratio=4)
    assert se(x).shape == (4, 16, 100)


def test_cbam_shape():
    x = torch.randn(4, 16, 100)
    cbam = CBAM1D(channels=16, reduction_ratio=4)
    assert cbam(x).shape == (4, 16, 100)


def test_attention_pooling_shape():
    x = torch.randn(4, 16, 100)
    pool = AttentionPooling1D(channels=16)
    assert pool(x).shape == (4, 16)


def test_bilstm_wrapper_shape():
    x = torch.randn(4, 16, 100)
    lstm = BiLSTMWrapper(in_channels=16, hidden_dim=32, num_layers=2)
    assert lstm(x).shape == (4, 64, 100) # output has 2 * hidden_dim channels
