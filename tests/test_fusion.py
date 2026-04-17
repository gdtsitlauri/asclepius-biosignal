"""Tests for Module 3 ASCLEPIUS-PULSE fusion model."""
import pytest
import torch

from asclepius.module3_fusion.fusion import ASCLEPIUSPulse


@pytest.fixture
def model():
    return ASCLEPIUSPulse(
        modalities=["eeg", "ecg", "ppg"],
        n_classes=3,
        d_model=64,
        n_heads=4,
        n_cma_layers=2,
        in_channels_override={"eeg": 4, "ecg": 1, "ppg": 1},
    )


def test_full_fusion_forward(model):
    inputs = {
        "eeg": torch.randn(2, 4, 128),
        "ecg": torch.randn(2, 1, 64),
        "ppg": torch.randn(2, 1, 64),
    }
    out = model(inputs)
    assert out.shape == (2, 3)


def test_missing_modality(model):
    inputs = {
        "eeg": torch.randn(2, 4, 128),
        "ecg": None,   # missing
        "ppg": torch.randn(2, 1, 64),
    }
    out = model(inputs)
    assert out.shape == (2, 3)


def test_uncertainty(model):
    inputs = {
        "eeg": torch.randn(1, 4, 128),
        "ecg": torch.randn(1, 1, 64),
        "ppg": torch.randn(1, 1, 64),
    }
    mean, uncertainty = model.predict_with_uncertainty(inputs, n_samples=5)
    assert mean.shape == (1, 3)
    assert uncertainty.shape == (1,)


def test_modality_importance(model):
    inputs = {
        "eeg": torch.randn(1, 4, 128),
        "ecg": torch.randn(1, 1, 64),
        "ppg": torch.randn(1, 1, 64),
    }
    importance = model.get_modality_importance(inputs)
    assert set(importance.keys()) == {"eeg", "ecg", "ppg"}
    total = sum(importance.values())
    assert abs(total - 1.0) < 0.05   # gates should sum to ~1
