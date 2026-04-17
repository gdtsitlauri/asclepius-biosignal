"""Module 5 — Real-time monitoring of biomedical signals.

Compatible with:
  - OpenBCI (via BrainFlow)
  - Polar H10 (ECG/HR via BLE)
  - Empatica E4 (EDA/PPG via LSL)
  - Synthetic stream (for testing/demo)

Architecture:
  - Ring buffer per channel
  - Background thread for streaming ingestion
  - On-window-complete: anomaly detection → prediction → alert
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from loguru import logger


# ── Ring Buffer ───────────────────────────────────────────────────────────────

class RingBuffer:
    """Thread-safe ring buffer for a single channel."""

    def __init__(self, maxlen: int):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray):
        with self._lock:
            self._buf.extend(samples.tolist())

    def get(self) -> np.ndarray:
        with self._lock:
            return np.array(list(self._buf))

    def __len__(self):
        with self._lock:
            return len(self._buf)


# ── Alert ─────────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    timestamp: float
    signal_type: str
    anomaly_score: float
    is_anomaly: bool
    explanation: str
    prediction: Optional[str] = None
    uncertainty: Optional[float] = None

    def __str__(self):
        tag = "🚨 ANOMALY" if self.is_anomaly else "✓ Normal"
        base = f"[{tag}] {self.signal_type.upper()} @ {self.timestamp:.1f}s | score={self.anomaly_score:.2f}"
        if self.explanation:
            base += f" | {self.explanation}"
        if self.prediction:
            base += f" | pred={self.prediction}"
        return base


# ── Stream source base ────────────────────────────────────────────────────────

class SignalStreamSource:
    """Abstract base for streaming signal sources."""

    def __init__(self, signal_type: str, n_channels: int, fs: float):
        self.signal_type = signal_type
        self.n_channels = n_channels
        self.fs = fs

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def read(self) -> Optional[np.ndarray]:
        """Return (n_channels, n_new_samples) or None if no new data."""
        raise NotImplementedError


class SyntheticStreamSource(SignalStreamSource):
    """Generates synthetic biomedical signals for testing."""

    SIGNAL_PARAMS = {
        "eeg": {"freq": 10.0, "noise": 15.0},
        "ecg": {"freq": 1.2, "noise": 0.1},
        "emg": {"freq": 50.0, "noise": 0.1},
        "eda": {"freq": 0.05, "noise": 0.5},
        "ppg": {"freq": 1.2, "noise": 0.05},
    }

    def __init__(self, signal_type: str, n_channels: int, fs: float, chunk_size: int = 32):
        super().__init__(signal_type, n_channels, fs)
        self.chunk_size = chunk_size
        self._t = 0.0
        self._running = False
        self._params = self.SIGNAL_PARAMS.get(signal_type, {"freq": 1.0, "noise": 0.1})

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def read(self) -> Optional[np.ndarray]:
        if not self._running:
            return None
        t = np.arange(self.chunk_size) / self.fs + self._t
        self._t += self.chunk_size / self.fs
        freq = self._params["freq"]
        noise = self._params["noise"]
        data = np.stack([
            np.sin(2 * np.pi * freq * t) * noise + np.random.randn(self.chunk_size) * noise * 0.1
            for _ in range(self.n_channels)
        ])
        time.sleep(self.chunk_size / self.fs)
        return data


class BrainFlowStreamSource(SignalStreamSource):
    """OpenBCI via BrainFlow SDK."""

    def __init__(self, signal_type: str, n_channels: int, fs: float,
                 board_id: int = 0, serial_port: str = ""):
        super().__init__(signal_type, n_channels, fs)
        self.board_id = board_id
        self.serial_port = serial_port
        self._board = None

    def start(self):
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams
            params = BrainFlowInputParams()
            params.serial_port = self.serial_port
            self._board = BoardShim(self.board_id, params)
            self._board.prepare_session()
            self._board.start_stream()
            logger.info(f"BrainFlow stream started (board_id={self.board_id})")
        except Exception as e:
            logger.error(f"BrainFlow start failed: {e}. Falling back to synthetic.")
            self._board = None

    def stop(self):
        if self._board:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception:
                pass

    def read(self) -> Optional[np.ndarray]:
        if self._board is None:
            return None
        try:
            from brainflow.board_shim import BoardShim
            data = self._board.get_board_data()
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            return data[eeg_channels]
        except Exception:
            return None


# ── Real-time Monitor ─────────────────────────────────────────────────────────

@dataclass
class RealTimeMonitor:
    """Real-time biomedical signal monitoring engine.

    Ingests from any SignalStreamSource, applies anomaly detection
    and optional prediction model on sliding windows.
    """

    signal_type: str
    n_channels: int
    sampling_rate: float
    window_seconds: float
    step_seconds: float = 0.5
    anomaly_detector: Optional[object] = None   # AsclepiusAnomalyDetector
    predictor: Optional[object] = None          # DiseasePredictorPipeline model
    alert_callbacks: List[Callable] = field(default_factory=list)
    alert_threshold: float = 2.5
    device: Optional[object] = None

    _running: bool = field(default=False, init=False, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _alert_queue: queue.Queue = field(default_factory=queue.Queue, init=False)
    _buffers: Dict[int, RingBuffer] = field(default_factory=dict, init=False)

    def __post_init__(self):
        win_samples = int(self.window_seconds * self.sampling_rate)
        self._buffers = {
            ch: RingBuffer(maxlen=win_samples * 3)
            for ch in range(self.n_channels)
        }
        self._win_samples = win_samples
        self._step_samples = int(self.step_seconds * self.sampling_rate)
        self._samples_since_last = 0

    def add_alert_callback(self, cb: Callable[[Alert], None]):
        self.alert_callbacks.append(cb)

    def start(self, source: SignalStreamSource):
        self._running = True
        source.start()
        self._thread = threading.Thread(
            target=self._run_loop, args=(source,), daemon=True
        )
        self._thread.start()
        logger.info(f"[RealTimeMonitor] Started monitoring {self.signal_type.upper()}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[RealTimeMonitor] Stopped.")

    def get_pending_alerts(self) -> List[Alert]:
        alerts = []
        while not self._alert_queue.empty():
            alerts.append(self._alert_queue.get_nowait())
        return alerts

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_loop(self, source: SignalStreamSource):
        while self._running:
            data = source.read()
            if data is None:
                time.sleep(0.01)
                continue
            # Push new samples to per-channel buffers
            for ch in range(min(self.n_channels, data.shape[0])):
                self._buffers[ch].push(data[ch])
            self._samples_since_last += data.shape[1]

            if self._samples_since_last >= self._step_samples:
                self._samples_since_last = 0
                self._process_window()

    def _process_window(self):
        # Check all buffers filled
        for ch, buf in self._buffers.items():
            if len(buf) < self._win_samples:
                return

        window = np.stack([
            self._buffers[ch].get()[-self._win_samples:]
            for ch in range(self.n_channels)
        ])  # (n_channels, win_samples)

        t = time.time()
        result = {"anomaly_score": 0.0, "is_anomaly": False, "explanation": ""}

        if self.anomaly_detector is not None:
            try:
                result = self.anomaly_detector.score(window)
            except Exception as e:
                logger.warning(f"Anomaly detection error: {e}")

        prediction = None
        uncertainty = None
        if self.predictor is not None and result.get("anomaly_score", 0) > self.alert_threshold:
            try:
                import torch
                x = torch.from_numpy(window[np.newaxis].astype(np.float32))
                if self.device:
                    x = x.to(self.device)
                with torch.no_grad():
                    logits = self.predictor(x)
                    pred_class = logits.argmax(-1).item()
                    prediction = str(pred_class)
            except Exception as e:
                logger.warning(f"Predictor error: {e}")

        alert = Alert(
            timestamp=t,
            signal_type=self.signal_type,
            anomaly_score=float(result.get("anomaly_score", 0)),
            is_anomaly=bool(result.get("is_anomaly", False)),
            explanation=result.get("explanation", ""),
            prediction=prediction,
            uncertainty=uncertainty,
        )

        self._alert_queue.put(alert)
        for cb in self.alert_callbacks:
            try:
                cb(alert)
            except Exception:
                pass

        if alert.is_anomaly:
            logger.warning(str(alert))
        else:
            logger.debug(str(alert))
