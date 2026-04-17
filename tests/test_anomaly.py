"""Tests for Module 2 anomaly detection."""
import numpy as np
import pytest
from asclepius.module2_anomaly.detector import AsclepiusAnomalyDetector


@pytest.mark.parametrize("sig", ["eeg", "ecg", "emg", "eda", "ppg"])
def test_anomaly_score(sig):
    fs_map = {"eeg": 256, "ecg": 360, "emg": 2000, "eda": 4, "ppg": 64}
    ch_map = {"eeg": 4, "ecg": 1, "emg": 4, "eda": 1, "ppg": 1}
    fs = fs_map[sig]
    n_ch = ch_map[sig]
    win = int(fs * 2)

    detector = AsclepiusAnomalyDetector(sig, float(fs))
    x = np.random.randn(n_ch, win).astype(np.float32)
    result = detector.score(x)

    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert "per_feature_zscore" in result
    assert "explanation" in result
    assert isinstance(result["anomaly_score"], float)


def test_fit_reference():
    detector = AsclepiusAnomalyDetector("ecg", 360.0)
    X_healthy = np.random.randn(20, 1, 360).astype(np.float32)
    detector.fit_reference(X_healthy)
    result = detector.score(X_healthy[0])
    assert result["anomaly_score"] < 5.0


def test_predict_batch():
    detector = AsclepiusAnomalyDetector("eeg", 256.0)
    X = np.random.randn(10, 4, 512).astype(np.float32)
    preds = detector.predict(X)
    assert preds.shape == (10,)
    assert set(preds).issubset({0, 1})


def test_unknown_signal_type():
    with pytest.raises(ValueError):
        AsclepiusAnomalyDetector("xyz", 256.0)
