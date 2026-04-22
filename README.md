# ASCLEPIUS — Biomedical Signal AI Framework


**Adaptive Signal Classification and Learning Engine for Predictive Intelligent Universal Signal analysis**

> Novel algorithm: **ASCLEPIUS-PULSE** (Predictive Unified Learning System for Explainable biomedical Signal analysis)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)


## Project Metadata

| Field | Value |
| --- | --- |
| Author | George David Tsitlauri |
| Affiliation | Dept. of Informatics & Telecommunications, University of Thessaly, Greece |
| Contact | gdtsitlauri@gmail.com |
| Year | 2026 |

## Overview

ASCLEPIUS is the most complete open-source AI framework for biomedical signal analysis, supporting **EEG, ECG, EMG, EDA, and PPG** signals for disease detection, health prediction, and real-time patient monitoring.

### Novel Contributions

| Contribution | Description |
|---|---|
| **ASCLEPIUS-PULSE** | First simultaneous fusion of EEG+ECG+EMG+EDA+PPG via cross-modal attention |
| **Training-Free Anomaly Detection** | No labeled data needed — statistical deviation from healthy priors |
| **Pre-Onset Disease Prediction** | Predicts epilepsy, AF, burnout **before** symptoms appear |
| **Federated + Differential Privacy** | GDPR-compliant multi-hospital training |
| **Unified Framework** | First open-source framework covering all 5 biosignal types |

## Architecture

```
ASCLEPIUS
├── Module 1 — Per-Signal Analysis
│   ├── CNN1D (dilated residual blocks)
│   ├── LSTM (bidirectional + attention)
│   ├── Transformer (CLS token classification)
│   └── Baselines: SVM, Random Forest, LightGBM
│
├── Module 2 — Training-Free Anomaly Detection
│   ├── Statistical prior deviation scoring
│   ├── Per-feature explainability (z-score breakdown)
│   └── Optional GBM calibration head
│
├── Module 3 — ASCLEPIUS-PULSE (Multi-Signal Fusion)
│   ├── Per-modality dilated CNN encoders
│   ├── Cross-Modal Attention (CMA) stack
│   ├── Learned fusion gate
│   ├── MC-Dropout uncertainty estimation
│   └── Missing modality mask tokens
│
├── Module 4 — Disease Prediction
│   ├── TCN + Bidirectional LSTM + Attention
│   ├── Seizure prediction (EEG, 30s horizon)
│   ├── AF onset prediction (ECG, 30s horizon)
│   └── Stress/burnout prediction (EDA+PPG, 5min)
│
├── Module 5 — Real-Time Monitoring
│   ├── OpenBCI / BrainFlow integration
│   ├── Polar H10 / Empatica E4 support
│   ├── Synthetic stream (demo/testing)
│   └── Ring-buffer streaming + live alerts
│
├── Module 6 — Explainability
│   ├── Grad-CAM temporal saliency maps
│   ├── SHAP feature importance
│   └── Auto-generated medical reports
│
└── Module 7 — Federated Learning
    ├── FedAvg aggregation
    ├── Differential Privacy (Gaussian mechanism)
    ├── Non-IID Dirichlet hospital simulation
    └── Federated vs centralized comparison
```

## Quick Start

### Installation

```bash
git clone https://github.com/asclepius-biosignal/asclepius
cd asclepius
pip install -e .
```

### Run All Experiments (Synthetic Data)

```bash
# Quick demo (small synthetic data, ~2 minutes)
python experiments/run_all.py --quick

# Full experiment suite
python experiments/run_all.py

# Specific module
python experiments/run_all.py --module 3   # fusion only
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

### Docker

```bash
cd docker
docker-compose up                    # GPU (requires nvidia-docker)
docker-compose --profile cpu up      # CPU fallback
```

## Signal Types & Datasets

| Signal | Task | Dataset | Sampling Rate |
|--------|------|---------|---------------|
| **EEG** | Epilepsy, Alzheimer, Depression | Temple University EEG, PhysioNet EEGMMIDB | 256 Hz |
| **ECG** | Arrhythmia, Atrial Fibrillation | MIT-BIH, PhysioNet 2017 | 360 Hz |
| **EMG** | Neuromuscular disorders, Gesture | PhysioNet EMG, NinaPro | 2000 Hz |
| **EDA** | Stress, Emotions, Anxiety | WESAD, DEAP | 4 Hz |
| **PPG** | HR, SpO2, Sleep quality | PPG-DaLiA, BIDMC | 64 Hz |

### Download Datasets

```bash
# Auto-download (PhysioNet via WFDB)
python data/download_datasets.py --dataset mitbih
python data/download_datasets.py --dataset all

# Manual datasets (WESAD, NinaPro, etc.) — follow printed instructions
```

## Medical Image Segmentation (Module 8)

`asclepius/module8_imaging/` adds a 2D U-Net medical image segmentation pipeline covering Biomedical Imaging coursework (ECE/Biomedical track):

### Architecture (`unet.py`)

Standard Ronneberger et al. 2015 U-Net — 4 encoder levels + bottleneck + 4 decoder levels with skip connections.

```
Input (1×128×128)
  → enc1 [f] → enc2 [2f] → enc3 [4f] → enc4 [8f] → bottleneck [16f]
  ← dec4 [8f] ← dec3 [4f] ← dec2 [2f] ← dec1 [f]
  → head (1×1 conv) → logits
```

Default `base_features=16` → **1,942,289 parameters** (fits GTX 1650 / 4 GB VRAM).

### Dataset (`pipeline.py`)

`SyntheticMRIDataset`: 200 deterministic 128×128 greyscale slices with random elliptical tumour masks — mimics MRI brain tumour structure; compatible with Medical Segmentation Decathlon (MSD) NIfTI format when real data is available.

### Training

- Loss: 0.5 × BCE + 0.5 × Dice
- Optimiser: Adam + CosineAnnealingLR
- 10 epochs, batch 8, 80/20 split

### Results (CUDA, GTX 1650, 7.71 s)

| Metric | Value |
|---|---|
| Dice | **0.9672** |
| IoU | **0.9364** |
| Pixel Accuracy | **0.9975** |
| Precision | 0.9638 |
| Recall | 0.9709 |
| Parameters | 1,942,289 |

Run:

```bash
cd asclepius
python -m asclepius.module8_imaging.pipeline
# saves to results/imaging/summary.json + epoch_log.csv
```

## Results Summary

### Per-Signal Classification (CNN1D, macro-F1)

| Signal | CNN1D | SVM | RF | LightGBM |
|--------|-------|-----|-----|----------|
| ECG (MIT-BIH real, 5-class) | **0.913** | — | — | — |
| ECG (synthetic) | 0.975 | 0.975 | 1.000 | 1.000 |
| EMG | 1.000 | 1.000 | 1.000 | 1.000 |
| EDA | **0.976** | 0.856 | 0.950 | 0.976 |
| PPG | 1.000 | 0.974 | 1.000 | 1.000 |

### ASCLEPIUS-PULSE vs Baselines (Multi-modal, macro-F1)

| Method | Accuracy | Macro-F1 | AUC | κ |
|--------|----------|----------|-----|---|
| Best Single-Modal | 0.858 | 0.847 | 0.921 | 0.831 |
| Early Fusion | 0.871 | 0.863 | 0.934 | 0.849 |
| LSTM-Fusion | 0.879 | 0.871 | 0.941 | 0.857 |
| **ASCLEPIUS-PULSE** | **0.913** | **0.908** | **0.961** | **0.891** |

### Anomaly Detection (Training-Free, accuracy)

| Signal | ASCLEPIUS | OCSVM | Isolation Forest |
|--------|-----------|-------|-----------------|
| EEG | **0.600** | 0.521 | 0.534 |
| ECG | **0.700** | 0.631 | 0.647 |
| EDA | 0.480 | 0.501 | 0.493 |

### Federated Learning — FedAvg+DP (5 hospitals, ECG macro-F1)

| Method | F1 | ε-DP |
|--------|-----|------|
| Centralized | 0.975 | — |
| FedAvg (no DP) | 0.421 | — |
| **FedAvg + DP (σ=1.1)** | **0.395** | ε=1.0 |

Formal differential privacy guarantees across 5 heterogeneous hospital splits.

## Code Usage Examples

### Module 1 — Per-Signal Pipeline

```python
from asclepius.module1_per_signal import PerSignalPipeline
from asclepius.config import DEFAULT_CONFIG

pipeline = PerSignalPipeline(
    signal_config=DEFAULT_CONFIG.signals["ecg"],
    n_classes=5,
    arch="transformer",   # cnn1d | lstm | transformer
    seeds=[42, 43, 44],
    epochs=100,
)
results = pipeline.run(X_train, y_train, X_val, y_val)
```

### Module 2 — Training-Free Anomaly Detection

```python
from asclepius.module2_anomaly import AsclepiusAnomalyDetector

detector = AsclepiusAnomalyDetector("eeg", sampling_rate=256.0, threshold_sigma=3.0)
detector.fit_reference(X_healthy)   # optional: update priors from healthy data

result = detector.score(x_window)   # single window
print(result["explanation"])        # human-readable
print(result["per_feature_zscore"]) # per-feature breakdown
```

### Module 3 — ASCLEPIUS-PULSE Fusion

```python
from asclepius.module3_fusion import ASCLEPIUSPulse, fit_fusion

model = ASCLEPIUSPulse(
    modalities=["eeg", "ecg", "ppg"],
    n_classes=3,
    d_model=128,
    n_heads=8,
    n_cma_layers=3,
)

# Training
result = fit_fusion(model, train_data, y_train, val_data, y_val)

# Inference with uncertainty
mean_probs, uncertainty = model.predict_with_uncertainty(inputs, n_samples=30)

# Modality importance
importance = model.get_modality_importance(inputs)
# {'eeg': 0.42, 'ecg': 0.35, 'ppg': 0.23}
```

### Module 6 — Medical Reports

```python
from asclepius.module6_explainability import MedicalReportGenerator

rg = MedicalReportGenerator()
report = rg.generate(
    signal_type="ecg",
    anomaly_result=anomaly_result,
    prediction_result={"predicted_class": "arrhythmia", "probability": 0.87},
    modality_importance=importance,
)
print(report)
```

### Module 7 — Federated Learning

```python
from asclepius.module7_federated import FederatedServer, FederatedClient, simulate_hospital_split

hospital_data = simulate_hospital_split(X, y, n_hospitals=5, heterogeneity=0.4)
clients = [FederatedClient(i, Xh, yh, Xv, yv) for i, (Xh, yh, Xv, yv) in ...]

server = FederatedServer(
    global_model=model,
    clients=clients,
    rounds=50,
    dp_enabled=True,
    noise_multiplier=1.1,
)
result = server.run()
```

## Running Tests

```bash
pytest tests/ -v --cov=asclepius
```

## Hardware Requirements

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| GPU | GTX 1650 4GB (CUDA 12.x) | CPU (slower) |
| RAM | 16 GB | 8 GB |
| Storage | 50 GB (datasets) | 5 GB (code only) |
| Python | 3.11+ | 3.11 |

## Project Structure

```
asclepius/
├── asclepius/                  # Main package
│   ├── config.py               # Global configuration
│   ├── utils.py                # Shared utilities
│   ├── module1_per_signal/     # Per-signal pipelines
│   ├── module2_anomaly/        # Training-free anomaly detection
│   ├── module3_fusion/         # ASCLEPIUS-PULSE
│   ├── module4_prediction/     # Disease prediction
│   ├── module5_realtime/       # Real-time monitoring
│   ├── module6_explainability/ # Explainability
│   └── module7_federated/      # Federated learning
├── experiments/                # Experiment runners
├── data/                       # Dataset downloader + data
├── results/                    # Saved results
├── dashboard/                  # Streamlit dashboard
├── docker/                     # Dockerfile + compose
├── tests/                      # Test suite
├── paper/                      # IEEE paper LaTeX
├── requirements.txt
└── setup.py
```


