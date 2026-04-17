"""Classical ML baselines: SVM, Random Forest, LightGBM."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _try_lgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        return None


def build_baseline(name: str, **kwargs) -> Any:
    if name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
                gamma="scale",
                probability=True,
                random_state=kwargs.get("seed", 42),
            )),
        ])
    if name == "rf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 300),
                max_depth=kwargs.get("max_depth", None),
                n_jobs=-1,
                random_state=kwargs.get("seed", 42),
            )),
        ])
    if name == "lgbm":
        LGBM = _try_lgbm()
        if LGBM is None:
            raise ImportError("lightgbm not installed. pip install lightgbm")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBM(
                n_estimators=kwargs.get("n_estimators", 500),
                learning_rate=kwargs.get("lr", 0.05),
                num_leaves=kwargs.get("num_leaves", 63),
                n_jobs=-1,
                random_state=kwargs.get("seed", 42),
                verbose=-1,
            )),
        ])
    raise ValueError(f"Unknown baseline: {name}. Choose svm, rf, lgbm.")


BASELINE_NAMES = ["svm", "rf", "lgbm"]
