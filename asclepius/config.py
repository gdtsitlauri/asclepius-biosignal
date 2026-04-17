"""Global configuration for ASCLEPIUS."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml


ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"


@dataclass
class SignalConfig:
    name: str
    sampling_rate: int
    n_channels: int
    window_seconds: float
    overlap: float = 0.5
    bandpass: Optional[tuple] = None


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class FederatedConfig:
    n_clients: int = 5
    rounds: int = 50
    local_epochs: int = 3
    fraction_fit: float = 0.8
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0


@dataclass
class AsclepiusConfig:
    # Per-signal configs
    signals: Dict[str, SignalConfig] = field(default_factory=lambda: {
        "eeg": SignalConfig(
            name="eeg", sampling_rate=256, n_channels=64,
            window_seconds=4.0, bandpass=(0.5, 45.0)
        ),
        "ecg": SignalConfig(
            name="ecg", sampling_rate=360, n_channels=1,
            window_seconds=10.0, bandpass=(0.5, 40.0)
        ),
        "emg": SignalConfig(
            name="emg", sampling_rate=2000, n_channels=8,
            window_seconds=1.0, bandpass=(20.0, 500.0)
        ),
        "eda": SignalConfig(
            name="eda", sampling_rate=4, n_channels=1,
            window_seconds=60.0
        ),
        "ppg": SignalConfig(
            name="ppg", sampling_rate=64, n_channels=1,
            window_seconds=30.0, bandpass=(0.5, 8.0)
        ),
    })
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    prediction_horizon_seconds: int = 30
    anomaly_threshold_sigma: float = 3.0
    results_dir: Path = RESULTS_DIR
    data_dir: Path = DATA_DIR

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AsclepiusConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        # shallow override training block
        if "training" in raw:
            for k, v in raw["training"].items():
                setattr(cfg.training, k, v)
        if "federated" in raw:
            for k, v in raw["federated"].items():
                setattr(cfg.federated, k, v)
        return cfg

    def get_device(self) -> torch.device:
        return torch.device(self.training.resolve_device())


DEFAULT_CONFIG = AsclepiusConfig()
