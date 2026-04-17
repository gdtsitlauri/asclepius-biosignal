"""Feature extraction: time-domain, frequency-domain, wavelet."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pywt
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew


# ── Time-domain features ──────────────────────────────────────────────────────

def time_domain_features(x: np.ndarray, fs: float) -> np.ndarray:
    """x: (n_channels, n_samples) → feature vector."""
    feats = []
    for ch in x:
        feats.extend([
            np.mean(ch),
            np.std(ch),
            np.var(ch),
            np.sqrt(np.mean(ch ** 2)),           # RMS
            np.max(np.abs(ch)),                   # peak amplitude
            skew(ch),
            kurtosis(ch),
            np.sum(np.abs(np.diff(ch))),          # mean absolute deviation
            np.mean(np.abs(np.diff(ch))),         # mean absolute difference
            zero_crossing_rate(ch),
            hjorth_mobility(ch),
            hjorth_complexity(ch),
        ])
    return np.array(feats, dtype=np.float32)


def zero_crossing_rate(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(np.sign(x)))) / (2 * len(x)))


def hjorth_mobility(x: np.ndarray) -> float:
    dx = np.diff(x)
    return float(np.std(dx) / (np.std(x) + 1e-8))


def hjorth_complexity(x: np.ndarray) -> float:
    return float(hjorth_mobility(np.diff(x)) / (hjorth_mobility(x) + 1e-8))


# ── Frequency-domain features ─────────────────────────────────────────────────

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def frequency_domain_features(
    x: np.ndarray,
    fs: float,
    bands: Optional[dict] = None,
) -> np.ndarray:
    """x: (n_channels, n_samples) → feature vector."""
    if bands is None:
        bands = EEG_BANDS
    feats = []
    for ch in x:
        freqs, psd = sp_signal.welch(ch, fs=fs, nperseg=min(256, len(ch)))
        total_power = np.trapezoid(psd, freqs) + 1e-8
        for lo, hi in bands.values():
            idx = (freqs >= lo) & (freqs <= hi)
            bp = np.trapezoid(psd[idx], freqs[idx])
            feats.extend([bp, bp / total_power])
        # spectral centroid & bandwidth
        sc = np.sum(freqs * psd) / (np.sum(psd) + 1e-8)
        sb = np.sqrt(np.sum((freqs - sc) ** 2 * psd) / (np.sum(psd) + 1e-8))
        feats.extend([sc, sb, np.max(psd), freqs[np.argmax(psd)]])
    return np.array(feats, dtype=np.float32)


# ── Wavelet features ──────────────────────────────────────────────────────────

def wavelet_features(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
) -> np.ndarray:
    """x: (n_channels, n_samples) → feature vector."""
    feats = []
    for ch in x:
        coeffs = pywt.wavedec(ch, wavelet, level=level)
        for c in coeffs:
            feats.extend([
                np.mean(np.abs(c)),
                np.std(c),
                np.sum(c ** 2),          # energy
                np.max(np.abs(c)),
            ])
    return np.array(feats, dtype=np.float32)


# ── Combined ──────────────────────────────────────────────────────────────────

def extract_all_features(
    x: np.ndarray,
    fs: float,
    bands: Optional[dict] = None,
    wavelet: str = "db4",
    wavelet_level: int = 5,
) -> np.ndarray:
    td = time_domain_features(x, fs)
    fd = frequency_domain_features(x, fs, bands)
    wd = wavelet_features(x, wavelet, wavelet_level)
    return np.concatenate([td, fd, wd])
