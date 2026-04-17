#!/usr/bin/env python3
"""Run all ASCLEPIUS experiments end-to-end with synthetic data.

Usage:
  python experiments/run_all.py                    # all modules
  python experiments/run_all.py --module 1         # per-signal only
  python experiments/run_all.py --module 3         # fusion only
  python experiments/run_all.py --quick            # small synthetic data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from asclepius.config import DEFAULT_CONFIG
from asclepius.utils import print_metrics_table, save_results, set_seed

console = Console()


# ── Synthetic data generators ─────────────────────────────────────────────────

def make_synthetic_signal(
    n_samples: int,
    n_classes: int,
    n_channels: int,
    window_size: int,
    seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_channels, window_size)).astype(np.float32)
    y = rng.integers(0, n_classes, n_samples)
    # Add class-discriminative signal
    for cls in range(n_classes):
        mask = y == cls
        freq = (cls + 1) * 2.0
        t = np.linspace(0, 1, window_size)
        X[mask] += np.sin(2 * np.pi * freq * t)[np.newaxis, np.newaxis]
    return X, y.astype(np.int64)


def make_multimodal_synthetic(n_samples: int, n_classes: int, quick: bool):
    configs = {
        "eeg": (4, 512),
        "ecg": (1, 256),
        "emg": (4, 128),
        "eda": (1, 64),
        "ppg": (1, 128),
    }
    if quick:
        configs = {k: (c, min(s, 128)) for k, (c, s) in configs.items()}

    data = {}
    for mod, (ch, win) in configs.items():
        X, _ = make_synthetic_signal(n_samples, n_classes, ch, win)
        data[mod] = X
    y = np.random.randint(0, n_classes, n_samples)
    return data, y.astype(np.int64)


# ── Module runners ─────────────────────────────────────────────────────────────

def run_module1(quick: bool = False):
    console.print(Panel("[bold cyan]Module 1 — Per-Signal Analysis[/bold cyan]"))
    from asclepius.module1_per_signal.pipeline import PerSignalPipeline
    from asclepius.config import DEFAULT_CONFIG

    N = 200 if quick else 1000
    results = {}

    signal_specs = [
        ("eeg", 4, 512, 2),
        ("ecg", 1, 256, 2),
        ("emg", 4, 128, 5),
        ("eda", 1, 64, 3),
        ("ppg", 1, 128, 2),
    ]

    for sig_name, ch, win, n_cls in signal_specs:
        logger.info(f"Running {sig_name.upper()} pipeline...")
        X, y = make_synthetic_signal(N, n_cls, ch, win)
        split = int(0.8 * N)
        X_tr, y_tr = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        sig_cfg = DEFAULT_CONFIG.signals[sig_name]
        # Adjust window to match synthetic data
        sig_cfg = type(sig_cfg)(
            name=sig_name,
            sampling_rate=sig_cfg.sampling_rate,
            n_channels=ch,
            window_seconds=win / sig_cfg.sampling_rate,
            bandpass=sig_cfg.bandpass,
        )

        pipeline = PerSignalPipeline(
            signal_config=sig_cfg,
            n_classes=n_cls,
            arch="cnn1d",
            seeds=[42] if quick else [42, 43, 44],
            epochs=5 if quick else 30,
        )
        res = pipeline.run(
            X_tr, y_tr, X_val, y_val,
            save_path=ROOT / "results" / "per_signal" / f"{sig_name}.json",
        )
        results[sig_name] = res
        console.print(f"  [green]{sig_name.upper()}[/green] DL f1_macro: "
                      f"{res['deep_learning'].get('f1_macro', (0,0))[0]:.4f}")

    return results


def run_module2(quick: bool = False):
    console.print(Panel("[bold cyan]Module 2 — Anomaly Detection (Training-Free)[/bold cyan]"))
    from asclepius.module2_anomaly.detector import AsclepiusAnomalyDetector

    results = {}
    for sig_name in ["eeg", "ecg", "eda"]:
        N = 50 if quick else 200
        ch = {"eeg": 4, "ecg": 1, "eda": 1}[sig_name]
        win = {"eeg": 512, "ecg": 256, "eda": 64}[sig_name]
        fs = {"eeg": 256, "ecg": 360, "eda": 4}[sig_name]

        X_healthy, _ = make_synthetic_signal(N, 1, ch, win)
        X_test, y_test = make_synthetic_signal(N, 2, ch, win)

        detector = AsclepiusAnomalyDetector(sig_name, float(fs), threshold_sigma=2.5)
        detector.fit_reference(X_healthy)
        preds = detector.predict(X_test)

        from asclepius.utils import compute_metrics
        m = compute_metrics(y_test, preds)
        results[sig_name] = m
        console.print(f"  [green]{sig_name.upper()}[/green] accuracy={m['accuracy']:.4f}")

    save_results(ROOT / "results" / "anomaly_detection" / "results.json", results)
    return results


def run_module3(quick: bool = False):
    console.print(Panel("[bold cyan]Module 3 — ASCLEPIUS-PULSE Fusion[/bold cyan]"))
    from asclepius.module3_fusion.fusion import ASCLEPIUSPulse
    from asclepius.module3_fusion.fusion_trainer import fit_fusion

    N = 100 if quick else 500
    n_cls = 3
    modalities = ["eeg", "ecg", "eda", "ppg"]

    train_data, y_train = make_multimodal_synthetic(int(N * 0.8), n_cls, quick)
    val_data, y_val = make_multimodal_synthetic(int(N * 0.2), n_cls, quick)

    # Match channel counts from synthetic data
    ch_override = {m: train_data[m].shape[1] for m in modalities}

    model = ASCLEPIUSPulse(
        modalities=modalities,
        n_classes=n_cls,
        d_model=64 if quick else 128,
        n_heads=4 if quick else 8,
        n_cma_layers=2 if quick else 3,
        in_channels_override=ch_override,
    )
    result = fit_fusion(
        model,
        {m: train_data[m] for m in modalities},
        y_train,
        {m: val_data[m] for m in modalities},
        y_val,
        epochs=5 if quick else 50,
        batch_size=16,
    )
    save_results(ROOT / "results" / "fusion" / "asclepius_pulse.json", result)
    console.print(f"  [green]ASCLEPIUS-PULSE[/green] f1_macro={result['metrics'].get('f1_macro', 0):.4f}")
    return result


def run_module7(quick: bool = False):
    console.print(Panel("[bold cyan]Module 7 — Federated Learning[/bold cyan]"))
    from asclepius.module7_federated.federated import (
        FederatedClient, FederatedServer, simulate_hospital_split,
    )
    from asclepius.module1_per_signal.models import build_model

    N = 300 if quick else 1500
    n_cls = 2
    ch, win = 4, 256
    X, y = make_synthetic_signal(N, n_cls, ch, win)
    split = int(0.85 * N)
    X_all, y_all = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    hospital_data = simulate_hospital_split(X_all, y_all, n_hospitals=5, heterogeneity=0.4)

    clients = []
    for i, (Xh, yh) in enumerate(hospital_data):
        vs = int(0.2 * len(yh))
        clients.append(FederatedClient(
            client_id=i,
            X_train=Xh[vs:], y_train=yh[vs:],
            X_val=Xh[:vs], y_val=yh[:vs],
            local_epochs=2 if quick else 3,
        ))

    global_model = build_model("cnn1d", in_channels=ch, n_classes=n_cls)
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        rounds=5 if quick else 30,
        dp_enabled=True,
    )
    result = server.run()
    save_results(ROOT / "results" / "federated" / "fedavg_dp.json", result)
    f1 = result["final_metrics"].get("f1_macro", (0, 0))[0]
    console.print(f"  [green]FedAvg+DP[/green] final f1_macro={f1:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="ASCLEPIUS — Run All Experiments")
    parser.add_argument("--module", type=int, default=0, help="Run specific module (1-7, 0=all)")
    parser.add_argument("--quick", action="store_true", help="Quick run with small data")
    args = parser.parse_args()

    console.print(Panel(
        "[bold magenta]ASCLEPIUS — Biomedical Signal AI Framework[/bold magenta]\n"
        "Novel algorithm: ASCLEPIUS-PULSE",
        title="Welcome",
    ))

    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"  Using device: [bold green]{DEVICE}[/bold green]")
    set_seed(42)

    if args.module in (0, 1):
        run_module1(args.quick)
    if args.module in (0, 2):
        run_module2(args.quick)
    if args.module in (0, 3):
        run_module3(args.quick)
    if args.module in (0, 7):
        run_module7(args.quick)

    console.print(Panel(
        "[bold green]All experiments complete![/bold green]\n"
        f"Results saved to: {ROOT / 'results'}",
    ))


if __name__ == "__main__":
    main()
