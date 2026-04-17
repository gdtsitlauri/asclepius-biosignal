"""Per-signal analysis pipeline: preprocessing → feature extraction → training."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from asclepius.config import SignalConfig
from asclepius.module1_per_signal.baselines import BASELINE_NAMES, build_baseline
from asclepius.module1_per_signal.features import extract_all_features
from asclepius.module1_per_signal.models import build_model
from asclepius.module1_per_signal.trainer import fit, predict
from asclepius.utils import (
    aggregate_metrics, bandpass_filter, compute_metrics,
    normalize_signal, save_results, set_seed, sliding_windows,
)


@dataclass
class PerSignalPipeline:
    """Full per-signal analysis pipeline for one signal type."""

    signal_config: SignalConfig
    n_classes: int
    arch: str = "cnn1d"              # cnn1d | lstm | transformer
    model_hidden: int = 128
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    patience: int = 15
    device: Optional[Any] = None

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """X: (n_windows, n_channels, n_samples)."""
        out = []
        cfg = self.signal_config
        for win in X:
            if cfg.bandpass is not None:
                win = bandpass_filter(win, *cfg.bandpass, cfg.sampling_rate)
            win = normalize_signal(win, method="zscore")
            out.append(win)
        return np.stack(out)

    # ── Feature extraction (for baselines) ────────────────────────────────────

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """X: (n_windows, n_channels, n_samples) → (n_windows, n_features)."""
        return np.stack([
            extract_all_features(w, self.signal_config.sampling_rate)
            for w in X
        ])

    # ── Deep learning ─────────────────────────────────────────────────────────

    def _build_model(self):
        cfg = self.signal_config
        return build_model(
            self.arch,
            in_channels=cfg.n_channels,
            n_classes=self.n_classes,
            hidden=self.model_hidden,
        )

    def run_deep_learning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        all_metrics = []
        for seed in self.seeds:
            model = self._build_model()
            result = fit(
                model, X_train, y_train, X_val, y_val,
                epochs=self.epochs, batch_size=self.batch_size,
                lr=self.lr, patience=self.patience,
                device=self.device, seed=seed,
            )
            all_metrics.append(result["metrics"])
            logger.info(f"[{self.signal_config.name.upper()}] seed={seed} "
                        f"val_f1={result['metrics'].get('f1_macro', 0):.4f}")
        return aggregate_metrics(all_metrics)

    # ── Baselines ─────────────────────────────────────────────────────────────

    def run_baselines(
        self,
        X_train_feats: np.ndarray,
        y_train: np.ndarray,
        X_val_feats: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Dict]:
        results = {}
        for name in BASELINE_NAMES:
            seed_metrics = []
            for seed in self.seeds:
                clf = build_baseline(name, seed=seed)
                try:
                    clf.fit(X_train_feats, y_train)
                    pred = clf.predict(X_val_feats)
                    prob = clf.predict_proba(X_val_feats)[:, 1] if self.n_classes == 2 else None
                    m = compute_metrics(y_val, pred, prob)
                    seed_metrics.append(m)
                except Exception as e:
                    logger.warning(f"Baseline {name} failed: {e}")
            if seed_metrics:
                results[name] = aggregate_metrics(seed_metrics)
        return results

    # ── Full experiment ───────────────────────────────────────────────────────

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> Dict:
        sig = self.signal_config.name.upper()
        logger.info(f"[{sig}] Preprocessing...")
        X_train_p = self.preprocess(X_train)
        X_val_p = self.preprocess(X_val)

        logger.info(f"[{sig}] Extracting features for baselines...")
        X_train_feats = self.extract_features(X_train_p)
        X_val_feats = self.extract_features(X_val_p)

        logger.info(f"[{sig}] Running deep learning ({self.arch})...")
        dl_metrics = self.run_deep_learning(X_train_p, y_train, X_val_p, y_val)

        logger.info(f"[{sig}] Running baselines...")
        baseline_metrics = self.run_baselines(X_train_feats, y_train, X_val_feats, y_val)

        results = {
            "signal": self.signal_config.name,
            "arch": self.arch,
            "deep_learning": dl_metrics,
            "baselines": baseline_metrics,
        }

        if save_path:
            save_results(save_path, results)

        return results
