"""Module 2 — Training-free anomaly detection for biomedical signals.

Each signal window is scored against a reference distribution of healthy
statistics.  No training data required — works purely from statistical
deviation.  An optional GBM head can be trained for calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.stats import median_abs_deviation, zscore


# ── Statistical reference profiles ───────────────────────────────────────────

# Typical healthy ranges per signal type (mean ± std assumptions)
HEALTHY_PRIORS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "eeg": {
        "mean_rms": (15.0, 8.0),           # µV
        "spectral_centroid": (12.0, 5.0),  # Hz
        "delta_ratio": (0.25, 0.10),
        "alpha_ratio": (0.15, 0.08),
        "hjorth_mobility": (0.5, 0.2),
    },
    "ecg": {
        "mean_rms": (0.5, 0.3),            # mV
        "heart_rate": (70.0, 15.0),        # bpm
        "hrv_rmssd": (40.0, 20.0),         # ms
        "qrs_duration": (90.0, 15.0),      # ms
        "qt_interval": (400.0, 40.0),      # ms
    },
    "emg": {
        "mean_rms": (0.1, 0.05),
        "median_frequency": (80.0, 30.0),  # Hz
        "zero_crossing_rate": (0.15, 0.08),
    },
    "eda": {
        "mean_amplitude": (5.0, 4.0),      # µS
        "scr_rate": (0.05, 0.04),          # peaks/s
        "tonic_level": (3.0, 2.5),
    },
    "ppg": {
        "heart_rate": (70.0, 15.0),
        "pulse_amplitude": (0.5, 0.25),
        "spo2_estimate": (97.5, 1.5),
    },
}


# ── Feature extractors for anomaly scoring ────────────────────────────────────

def _extract_anomaly_features(
    x: np.ndarray,
    fs: float,
    signal_type: str,
) -> Dict[str, float]:
    """Compute interpretable scalar features from one signal window."""
    from scipy.signal import welch
    from scipy.stats import kurtosis

    if x.ndim == 1:
        x = x[np.newaxis, :]

    ch0 = x[0]
    freqs, psd = welch(ch0, fs=fs, nperseg=min(256, len(ch0)))
    total_power = np.trapezoid(psd, freqs) + 1e-10

    feats: Dict[str, float] = {}

    # Common
    feats["mean_rms"] = float(np.sqrt(np.mean(ch0 ** 2)))
    feats["kurtosis"] = float(kurtosis(ch0))
    feats["zero_crossing_rate"] = float(
        np.sum(np.abs(np.diff(np.sign(ch0)))) / (2 * len(ch0))
    )

    sc = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))
    feats["spectral_centroid"] = sc

    if signal_type == "eeg":
        for band, (lo, hi) in [
            ("delta", (0.5, 4)), ("theta", (4, 8)),
            ("alpha", (8, 13)), ("beta", (13, 30)), ("gamma", (30, 45)),
        ]:
            idx = (freqs >= lo) & (freqs <= hi)
            feats[f"{band}_ratio"] = float(np.trapezoid(psd[idx], freqs[idx]) / total_power)
        dx = np.diff(ch0)
        feats["hjorth_mobility"] = float(np.std(dx) / (np.std(ch0) + 1e-10))

    elif signal_type == "ecg":
        # Rough R-peak detection via local maxima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(ch0, distance=int(fs * 0.4), height=np.std(ch0) * 0.5)
        if len(peaks) >= 2:
            rr_ms = np.diff(peaks) / fs * 1000
            feats["heart_rate"] = float(60000 / np.mean(rr_ms))
            feats["hrv_rmssd"] = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
        else:
            feats["heart_rate"] = 70.0
            feats["hrv_rmssd"] = 40.0
        feats["qrs_duration"] = 90.0   # placeholder; real impl needs QRS delineation
        feats["qt_interval"] = 400.0

    elif signal_type == "emg":
        feats["median_frequency"] = float(
            freqs[np.searchsorted(np.cumsum(psd), np.sum(psd) / 2)]
        )

    elif signal_type == "eda":
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(ch0, prominence=0.05)
        win_s = len(ch0) / fs
        feats["mean_amplitude"] = float(np.mean(np.abs(ch0)))
        feats["scr_rate"] = float(len(peaks) / win_s)
        feats["tonic_level"] = float(np.percentile(ch0, 10))

    elif signal_type == "ppg":
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(ch0, distance=int(fs * 0.4))
        if len(peaks) >= 2:
            rr = np.diff(peaks) / fs
            feats["heart_rate"] = float(60 / np.mean(rr))
        else:
            feats["heart_rate"] = 70.0
        feats["pulse_amplitude"] = float(np.ptp(ch0))
        # Crude SpO2 estimate (requires red/IR channels; use placeholder)
        feats["spo2_estimate"] = 97.5

    return feats


# ── Anomaly Detector ──────────────────────────────────────────────────────────

@dataclass
class AsclepiusAnomalyDetector:
    """Training-free anomaly detector with per-feature score breakdown.

    Works by comparing extracted features against prior healthy distributions.
    An optional lightweight GBM head can be trained on labelled data to
    recalibrate scores.
    """

    signal_type: str                # eeg | ecg | emg | eda | ppg
    sampling_rate: float
    threshold_sigma: float = 3.0    # deviation in σ to flag anomaly
    use_gbm_head: bool = False
    _gbm: Optional[object] = field(default=None, init=False, repr=False)
    _reference: Dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        sig = self.signal_type.lower()
        if sig not in HEALTHY_PRIORS:
            raise ValueError(f"Unknown signal type '{sig}'. Choose from {list(HEALTHY_PRIORS)}")
        self._reference = HEALTHY_PRIORS[sig].copy()

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_reference(self, X_healthy: np.ndarray) -> "AsclepiusAnomalyDetector":
        """Update healthy reference stats from a sample of healthy windows.

        X_healthy: (n_windows, n_channels, n_samples)
        """
        all_feats = [
            _extract_anomaly_features(w, self.sampling_rate, self.signal_type)
            for w in X_healthy
        ]
        feat_keys = list(all_feats[0].keys())
        for k in feat_keys:
            vals = np.array([f[k] for f in all_feats])
            self._reference[k] = (float(np.median(vals)), float(median_abs_deviation(vals) + 1e-8))
        logger.info(f"[AnomalyDetector] Reference updated from {len(X_healthy)} healthy windows.")
        return self

    def fit_gbm_head(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "AsclepiusAnomalyDetector":
        """Optional: train a GBM classifier on labelled data."""
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("lightgbm required for GBM head.")
        scores = np.array([self._score_window(w) for w in X_train])
        feats_arr = np.stack([list(s["per_feature_zscore"].values()) for s in scores])
        self._gbm = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        self._gbm.fit(feats_arr, y_train)
        logger.info("[AnomalyDetector] GBM head trained.")
        return self

    def score(self, x: np.ndarray) -> Dict:
        """Score a single window. Returns anomaly score and breakdown."""
        return self._score_window(x)

    def score_batch(self, X: np.ndarray) -> List[Dict]:
        """Score batch of windows. X: (n_windows, n_channels, n_samples)."""
        return [self._score_window(w) for w in X]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels (0=normal, 1=anomaly)."""
        results = self.score_batch(X)
        if self._gbm is not None:
            feats_arr = np.stack([list(r["per_feature_zscore"].values()) for r in results])
            return self._gbm.predict(feats_arr)
        return np.array([r["is_anomaly"] for r in results], dtype=int)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score_window(self, x: np.ndarray) -> Dict:
        feats = _extract_anomaly_features(x, self.sampling_rate, self.signal_type)
        per_feature_zscore: Dict[str, float] = {}
        for k, val in feats.items():
            if not np.isfinite(val):
                per_feature_zscore[k] = 0.0
                continue
            if k in self._reference:
                mu, sigma = self._reference[k]
                z = abs(val - mu) / (sigma + 1e-10)
                per_feature_zscore[k] = float(z) if np.isfinite(z) else 0.0
            else:
                per_feature_zscore[k] = 0.0

        if per_feature_zscore:
            composite = float(np.nanmax(list(per_feature_zscore.values())))
        else:
            composite = 0.0

        is_anomaly = composite > self.threshold_sigma
        top_features = sorted(per_feature_zscore.items(), key=lambda x: -x[1])[:3]

        return {
            "anomaly_score": composite,
            "is_anomaly": bool(is_anomaly),
            "per_feature_zscore": per_feature_zscore,
            "raw_features": feats,
            "top_anomalous_features": top_features,
            "explanation": self._explain(top_features, is_anomaly),
        }

    def _explain(self, top_features: list, is_anomaly: bool) -> str:
        if not is_anomaly:
            return "Signal within normal healthy range."
        parts = []
        for feat, score in top_features:
            parts.append(f"{feat} deviated {score:.1f}σ from healthy baseline")
        return "ANOMALY: " + "; ".join(parts) + "."
