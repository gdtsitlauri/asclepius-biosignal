"""Real dataset loaders for ASCLEPIUS.

Each loader returns (X, y) where:
  X: (n_windows, n_channels, window_size) float32
  y: (n_windows,) int64
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

DATA_DIR = Path(__file__).parent


# ── MIT-BIH Arrhythmia (ECG) ──────────────────────────────────────────────────
# 5-class AAMI standard: N, S, V, F, Q

MITBIH_CLASS_MAP = {
    # Normal
    'N': 0, '.': 0, 'e': 0, 'j': 0,
    # Supra-ventricular ectopic
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    # Ventricular ectopic
    'V': 2, 'E': 2,
    # Fusion
    'F': 3,
    # Unknown / pacemaker
    'Q': 4, '/': 4, 'f': 4, 'u': 4,
}

MITBIH_LABELS = ['Normal', 'Supra-V', 'Ventricular', 'Fusion', 'Unknown']


def load_mitbih(
    data_dir: Optional[Path] = None,
    window_before: int = 90,
    window_after: int = 90,
    max_records: Optional[int] = None,
    balance: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load MIT-BIH Arrhythmia dataset.

    Returns windows centred on each R-peak annotation.
    X: (N, 1, window_before+window_after) float32
    y: (N,) int64  — 5 AAMI classes
    """
    import wfdb

    data_dir = data_dir or DATA_DIR / "ecg" / "mitbih"
    hea_files = sorted(data_dir.glob("*.hea"))
    if max_records:
        hea_files = hea_files[:max_records]

    X_all, y_all = [], []
    win = window_before + window_after

    for hea in hea_files:
        rec_name = str(hea.with_suffix(""))
        try:
            record = wfdb.rdrecord(rec_name, channels=[0])
            ann = wfdb.rdann(rec_name, "atr")
        except Exception as e:
            logger.warning(f"Skipping {hea.name}: {e}")
            continue

        sig = record.p_signal[:, 0].astype(np.float32)
        # z-score normalise
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)

        for idx, sym in zip(ann.sample, ann.symbol):
            if sym not in MITBIH_CLASS_MAP:
                continue
            start = idx - window_before
            end = idx + window_after
            if start < 0 or end > len(sig):
                continue
            X_all.append(sig[start:end][np.newaxis])   # (1, win)
            y_all.append(MITBIH_CLASS_MAP[sym])

    X = np.stack(X_all).astype(np.float32)
    y = np.array(y_all, dtype=np.int64)
    logger.info(f"MIT-BIH: {len(X)} beats from {len(hea_files)} records, "
                f"classes={dict(zip(*np.unique(y, return_counts=True)))}")

    if balance:
        X, y = _balance(X, y, seed=seed)

    return X, y


# ── BIDMC PPG ─────────────────────────────────────────────────────────────────
# Binary task: normal HR (60-100 bpm) vs. abnormal

def load_bidmc(
    data_dir: Optional[Path] = None,
    fs: float = 125.0,
    window_seconds: float = 30.0,
    overlap: float = 0.5,
    max_records: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load BIDMC PPG dataset.

    Labels: 0 = normal HR (60-100 bpm), 1 = abnormal HR
    X: (N, 1, window_size) float32
    """
    import wfdb

    data_dir = data_dir or DATA_DIR / "ppg" / "bidmc"
    hea_files = sorted(data_dir.glob("*.hea"))
    if max_records:
        hea_files = hea_files[:max_records]

    win = int(window_seconds * fs)
    step = int(win * (1 - overlap))

    X_all, y_all = [], []

    for hea in hea_files:
        rec_name = str(hea.with_suffix(""))
        try:
            record = wfdb.rdrecord(rec_name)
        except Exception as e:
            logger.warning(f"Skipping {hea.name}: {e}")
            continue

        # PPG is channel 0, ECG channel 1, respiration channel 2
        # Use PPG signal
        sig_names = [s.lower() for s in record.sig_name]
        ppg_idx = next((i for i, s in enumerate(sig_names) if 'ppg' in s or 'pleth' in s), 0)
        sig = record.p_signal[:, ppg_idx].astype(np.float32)
        sig = np.nan_to_num(sig)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)

        # Estimate HR from PPG peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(sig, distance=int(fs * 0.4))

        # Sliding windows
        n_total = len(sig)
        for start in range(0, n_total - win + 1, step):
            window = sig[start:start + win]
            # HR for this window
            w_peaks = peaks[(peaks >= start) & (peaks < start + win)]
            if len(w_peaks) >= 2:
                rr = np.diff(w_peaks) / fs
                hr = 60.0 / np.mean(rr)
                label = 0 if 60 <= hr <= 100 else 1
            else:
                label = 0
            X_all.append(window[np.newaxis])
            y_all.append(label)

    X = np.stack(X_all).astype(np.float32)
    y = np.array(y_all, dtype=np.int64)
    logger.info(f"BIDMC: {len(X)} windows from {len(hea_files)} records, "
                f"classes={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# ── PhysioNet EEG Motor Imagery ───────────────────────────────────────────────

def load_physionet_eeg(
    data_dir: Optional[Path] = None,
    fs: float = 160.0,
    window_seconds: float = 4.0,
    max_subjects: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load PhysioNet EEG Motor Movement/Imagery dataset.

    Binary: rest (0) vs. motor imagery (1)
    """
    import wfdb, mne

    data_dir = data_dir or DATA_DIR / "eeg" / "physionet_eeg"
    edf_files = sorted(data_dir.rglob("*.edf"))

    if not edf_files:
        raise FileNotFoundError(
            f"No EDF files found in {data_dir}. "
            "Run: python3 data/download_datasets.py --dataset physionet_eeg"
        )

    if max_subjects:
        # Group by subject (S001, S002...)
        subjects = sorted(set(f.parent for f in edf_files))[:max_subjects]
        edf_files = [f for f in edf_files if f.parent in subjects]

    win = int(window_seconds * fs)
    X_all, y_all = [], []

    for edf in edf_files:
        try:
            raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            data = raw.get_data().astype(np.float32)   # (n_ch, n_times)

            # z-score per channel
            data = (data - data.mean(1, keepdims=True)) / (data.std(1, keepdims=True) + 1e-8)

            n_ch, n_total = data.shape
            n_ch = min(n_ch, 64)
            data = data[:n_ch]

            for ev in events:
                start = ev[0]
                label = 0 if ev[2] == 1 else 1  # 1=rest, else=imagery
                if start + win > n_total:
                    continue
                X_all.append(data[:, start:start + win])
                y_all.append(label)
        except Exception as e:
            logger.warning(f"Skipping {edf.name}: {e}")

    if not X_all:
        raise ValueError("No EEG windows extracted.")

    X = np.stack(X_all).astype(np.float32)
    y = np.array(y_all, dtype=np.int64)
    logger.info(f"PhysioNet EEG: {len(X)} windows, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# ── Helpers ───────────────────────────────────────────────────────────────────

def _balance(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Undersample majority classes to match minority."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    idx = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        idx.append(rng.choice(cls_idx, size=min_count, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple:
    from sklearn.model_selection import train_test_split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_ratio + test_ratio,
        stratify=y, random_state=seed,
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1 - val_frac,
        stratify=y_tmp, random_state=seed,
    )
    return X_tr, y_tr, X_val, y_val, X_test, y_test
