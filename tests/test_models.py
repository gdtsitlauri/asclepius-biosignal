"""Tests for Module 1 deep learning models."""
import pytest
import torch
import numpy as np

from asclepius.module1_per_signal.models import CNN1DModel, LSTMModel, TransformerModel, build_model


@pytest.mark.parametrize("arch,cls", [
    ("cnn1d", CNN1DModel),
    ("lstm", LSTMModel),
    ("transformer", TransformerModel),
])
def test_model_forward(arch, cls):
    B, C, T = 4, 4, 256
    n_classes = 3
    model = build_model(arch, in_channels=C, n_classes=n_classes)
    x = torch.randn(B, C, T)
    out = model(x)
    assert out.shape == (B, n_classes), f"{arch} output shape mismatch"


def test_model_encode():
    model = CNN1DModel(in_channels=4, n_classes=2, hidden=64)
    x = torch.randn(2, 4, 128)
    enc = model.encode(x)
    assert enc.shape == (2, 64)


def test_build_model_unknown():
    with pytest.raises(ValueError):
        build_model("unknown", in_channels=1, n_classes=2)
