"""Module 4 — Disease Prediction (pre-symptom onset).

Predicts diseases BEFORE symptoms appear using temporal context windows.
Supported tasks:
  - Epileptic seizure prediction (EEG)
  - Atrial fibrillation onset (ECG)
  - Stress/burnout prediction (EDA + PPG)

Architecture: Temporal Convolutional Network + LSTM with configurable
prediction horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from asclepius.utils import compute_metrics, set_seed


# ── Temporal Prediction Model ─────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    """Dilated causal conv block (TCN-style)."""

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=pad, dilation=dilation)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=pad, dilation=dilation)
        )
        self.crop = pad
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

    def forward(self, x):
        h = self.drop(self.act(self.conv1(x)[..., : -self.crop or None]))
        h = self.drop(self.act(self.conv2(h)[..., : -self.crop or None]))
        res = x if self.downsample is None else self.downsample(x)
        return self.act(h + res)


class DiseasePredictorNet(nn.Module):
    """TCN + LSTM for pre-onset disease prediction."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        tcn_channels: int = 64,
        n_tcn_layers: int = 6,
        lstm_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        tcn_layers = []
        ch_in = in_channels
        for i in range(n_tcn_layers):
            tcn_layers.append(
                TemporalBlock(ch_in, tcn_channels, kernel_size=3,
                               dilation=2 ** i, dropout=dropout)
            )
            ch_in = tcn_channels
        self.tcn = nn.Sequential(*tcn_layers)
        self.lstm = nn.LSTM(
            tcn_channels, lstm_hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.attn = nn.Linear(lstm_hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.tcn(x)                     # (B, tcn_channels, T)
        h = h.permute(0, 2, 1)             # (B, T, tcn_channels)
        out, _ = self.lstm(h)
        scores = self.attn(out).squeeze(-1)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (out * w).sum(1)
        return self.head(ctx)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x).permute(0, 2, 1)
        out, _ = self.lstm(h)
        scores = self.attn(out).squeeze(-1)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (out * w).sum(1)


# ── Pre-onset dataset builder ─────────────────────────────────────────────────

def build_prediction_dataset(
    signal: np.ndarray,        # (n_channels, n_samples)
    events: np.ndarray,        # (n_events,) sample indices of onset
    fs: float,
    window_seconds: float = 30.0,
    horizon_seconds: float = 30.0,
    step_seconds: float = 5.0,
    negative_gap_seconds: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build pre-onset (positive) and non-pre-onset (negative) windows.

    Positive: windows that end `horizon_seconds` before onset.
    Negative: windows that start `negative_gap_seconds` after onset.
    """
    win = int(window_seconds * fs)
    horizon = int(horizon_seconds * fs)
    step = int(step_seconds * fs)
    neg_gap = int(negative_gap_seconds * fs)
    n_total = signal.shape[-1]

    X_pos, X_neg = [], []

    for onset in events:
        # Positive windows: [onset - horizon - win ... onset - horizon]
        end = onset - horizon
        start_range = max(0, end - win * 5)
        for s in range(start_range, end - win + 1, step):
            seg = signal[..., s:s + win]
            if seg.shape[-1] == win:
                X_pos.append(seg)

        # Negative windows after negative gap
        neg_start = onset + neg_gap
        for s in range(neg_start, min(n_total - win, neg_start + win * 5), step):
            seg = signal[..., s:s + win]
            if seg.shape[-1] == win:
                X_neg.append(seg)

    if not X_pos:
        raise ValueError("No positive windows found. Check events and signal length.")

    X = np.stack(X_pos + X_neg)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    return X, y


# ── Predictor pipeline ────────────────────────────────────────────────────────

@dataclass
class DiseasePredictorPipeline:
    """Pre-onset disease prediction pipeline."""

    task: str                    # seizure_prediction | af_onset | stress_burnout
    signal_type: str             # eeg | ecg | eda | ppg
    n_channels: int
    sampling_rate: float
    prediction_horizon_seconds: float = 30.0
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    epochs: int = 80
    batch_size: int = 16
    lr: float = 5e-4
    device: Optional[torch.device] = None

    TASK_CLASSES = {
        "seizure_prediction": 2,
        "af_onset": 2,
        "stress_burnout": 3,
    }

    def _n_classes(self):
        return self.TASK_CLASSES.get(self.task, 2)

    def _build_model(self):
        return DiseasePredictorNet(
            in_channels=self.n_channels,
            n_classes=self._n_classes(),
        )

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        from asclepius.module1_per_signal.trainer import fit, eval_epoch, make_loaders
        import torch.nn as nn

        all_metrics = []
        for seed in self.seeds:
            model = self._build_model()
            result = fit(
                model, X_train, y_train, X_val, y_val,
                epochs=self.epochs, batch_size=self.batch_size,
                lr=self.lr, patience=12, device=self.device, seed=seed,
            )
            all_metrics.append(result["metrics"])
            logger.info(
                f"[{self.task}] seed={seed} "
                f"f1={result['metrics'].get('f1_macro', 0):.4f} "
                f"auc={result['metrics'].get('auc', 0):.4f}"
            )
        from asclepius.utils import aggregate_metrics
        return {
            "task": self.task,
            "horizon_seconds": self.prediction_horizon_seconds,
            "metrics": aggregate_metrics(all_metrics),
        }
