"""Shared utilities for ASCLEPIUS."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        precision_score, recall_score, cohen_kappa_score,
    )
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return metrics


def aggregate_metrics(results: list[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Return mean ± std across seeds."""
    keys = results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[k] = (float(np.mean(vals)), float(np.std(vals)))
    return agg


def print_metrics_table(
    title: str,
    metrics: Dict[str, Tuple[float, float]],
) -> None:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    for k, (mean, std) in metrics.items():
        table.add_row(k, f"{mean:.4f}", f"{std:.4f}")
    console.print(table)


def save_results(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Results saved → {path}")


# ── Signal helpers ────────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    low: float,
    high: float,
    fs: float,
    order: int = 5,
) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    if signal.ndim == 1:
        return sosfilt(sos, signal)
    return np.stack([sosfilt(sos, ch) for ch in signal])


def normalize_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "zscore":
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True) + 1e-8
        return (signal - mean) / std
    if method == "minmax":
        mn = signal.min(axis=-1, keepdims=True)
        mx = signal.max(axis=-1, keepdims=True)
        return (signal - mn) / (mx - mn + 1e-8)
    raise ValueError(f"Unknown normalization method: {method}")


def sliding_windows(
    signal: np.ndarray,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """Return array of shape (n_windows, *signal.shape[:-1], window_size)."""
    length = signal.shape[-1]
    starts = range(0, length - window_size + 1, step_size)
    windows = [signal[..., s:s + window_size] for s in starts]
    return np.stack(windows)


# ── Timer ─────────────────────────────────────────────────────────────────────

class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self):
        return f"{self.elapsed:.3f}s"
