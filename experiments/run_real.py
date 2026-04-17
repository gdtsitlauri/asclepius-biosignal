#!/usr/bin/env python3
"""Full experiment run on real datasets.

Uses real data where available, synthetic fallback otherwise.
Run: python3 experiments/run_real.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from asclepius.config import DEFAULT_CONFIG
from asclepius.utils import aggregate_metrics, print_metrics_table, save_results, set_seed

console = Console()


# ── ECG — MIT-BIH ─────────────────────────────────────────────────────────────

def run_ecg_mitbih():
    console.print(Panel("[bold cyan]ECG — MIT-BIH Arrhythmia (5-class AAMI)[/bold cyan]"))
    from data.loaders import load_mitbih, train_val_test_split
    from asclepius.module1_per_signal.pipeline import PerSignalPipeline
    from asclepius.config import SignalConfig

    X, y = load_mitbih(balance=True)
    X_tr, y_tr, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    console.print(f"  Train={len(y_tr)} Val={len(y_val)} Test={len(y_test)}")

    sig_cfg = SignalConfig(
        name="ecg", sampling_rate=360, n_channels=1,
        window_seconds=(90 + 90) / 360, bandpass=(0.5, 40.0),
    )

    results = {}
    for arch in ["cnn1d", "lstm", "transformer"]:
        pipeline = PerSignalPipeline(
            signal_config=sig_cfg, n_classes=5,
            arch=arch, model_hidden=128,
            seeds=[42, 43, 44], epochs=80, batch_size=64, lr=1e-3,
            device=DEVICE,
        )
        res = pipeline.run(X_tr, y_tr, X_val, y_val)
        results[arch] = res["deep_learning"]
        f1 = res["deep_learning"].get("f1_macro", (0, 0))[0]
        console.print(f"  [green]{arch}[/green] val_f1={f1:.4f}")

    # Baselines on test set
    from asclepius.module1_per_signal.pipeline import PerSignalPipeline
    pipeline = PerSignalPipeline(sig_cfg, n_classes=5, arch="cnn1d", seeds=[42], device=DEVICE)
    X_tr_p = pipeline.preprocess(X_tr)
    X_test_p = pipeline.preprocess(X_test)
    X_tr_f = pipeline.extract_features(X_tr_p)
    X_test_f = pipeline.extract_features(X_test_p)
    baseline_res = pipeline.run_baselines(X_tr_f, y_tr, X_test_f, y_test)
    results["baselines"] = baseline_res

    save_results(ROOT / "results" / "per_signal" / "ecg_mitbih_real.json", results)
    return results


# ── PPG — BIDMC ───────────────────────────────────────────────────────────────

def run_ppg_bidmc():
    console.print(Panel("[bold cyan]PPG — BIDMC (normal vs abnormal HR)[/bold cyan]"))
    from data.loaders import load_bidmc, train_val_test_split
    from asclepius.module1_per_signal.pipeline import PerSignalPipeline
    from asclepius.config import SignalConfig

    X, y = load_bidmc(window_seconds=30.0)
    X_tr, y_tr, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    console.print(f"  Train={len(y_tr)} Val={len(y_val)} Test={len(y_test)}")

    sig_cfg = SignalConfig(
        name="ppg", sampling_rate=125, n_channels=1,
        window_seconds=30.0, bandpass=(0.5, 8.0),
    )

    results = {}
    for arch in ["cnn1d", "lstm", "transformer"]:
        pipeline = PerSignalPipeline(
            signal_config=sig_cfg, n_classes=2,
            arch=arch, model_hidden=128,
            seeds=[42, 43, 44], epochs=60, batch_size=32, lr=1e-3,
            device=DEVICE,
        )
        res = pipeline.run(X_tr, y_tr, X_val, y_val)
        results[arch] = res["deep_learning"]
        f1 = res["deep_learning"].get("f1_macro", (0, 0))[0]
        auc = res["deep_learning"].get("auc", (0, 0))[0]
        console.print(f"  [green]{arch}[/green] val_f1={f1:.4f} auc={auc:.4f}")

    save_results(ROOT / "results" / "per_signal" / "ppg_bidmc_real.json", results)
    return results


# ── Anomaly Detection on real ECG ─────────────────────────────────────────────

def run_anomaly_ecg():
    console.print(Panel("[bold cyan]Module 2 — Anomaly Detection on MIT-BIH ECG[/bold cyan]"))
    from data.loaders import load_mitbih
    from asclepius.module2_anomaly.detector import AsclepiusAnomalyDetector
    from asclepius.utils import compute_metrics

    X, y = load_mitbih(balance=False)   # use raw imbalanced data
    # Normal beats = class 0
    X_normal = X[y == 0]
    X_test = X
    y_binary = (y != 0).astype(int)    # 0=normal, 1=arrhythmia

    split = int(0.7 * len(X_normal))
    X_ref = X_normal[:split]

    detector = AsclepiusAnomalyDetector("ecg", 360.0, threshold_sigma=3.0)
    detector.fit_reference(X_ref)

    preds = detector.predict(X_test)
    metrics = compute_metrics(y_binary, preds)
    console.print(f"  accuracy={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}")

    save_results(ROOT / "results" / "anomaly_detection" / "ecg_mitbih_real.json", metrics)
    return metrics


# ── Fusion: ECG + PPG (ASCLEPIUS-PULSE) ───────────────────────────────────────

def run_fusion_ecg_ppg():
    console.print(Panel("[bold cyan]Module 3 — ASCLEPIUS-PULSE: ECG + PPG Fusion[/bold cyan]"))
    from data.loaders import load_mitbih, load_bidmc, train_val_test_split
    from asclepius.module3_fusion.fusion import ASCLEPIUSPulse
    from asclepius.module3_fusion.fusion_trainer import fit_fusion

    # Load ECG — binary (normal vs arrhythmia)
    X_ecg, y_ecg = load_mitbih(balance=True)
    X_ecg_tr, y_ecg_tr, X_ecg_val, _, _, _ = train_val_test_split(X_ecg, y_ecg)
    # Trim to 2 classes: normal (0) vs arrhythmia (1+)
    y_ecg_tr = (y_ecg_tr > 0).astype(np.int64)
    y_ecg_val = ((_ > 0) if False else y_ecg_val).astype(np.int64) if False else \
                (y_ecg_val > 0).astype(np.int64) if hasattr(y_ecg_val := train_val_test_split(X_ecg, y_ecg)[2], '__len__') else None

    # Simpler: just reload
    X_ecg, y_ecg = load_mitbih(balance=True)
    y_ecg = (y_ecg > 0).astype(np.int64)  # binary
    from sklearn.model_selection import train_test_split
    X_ecg_tr, X_ecg_val, y_tr, y_val = train_test_split(
        X_ecg, y_ecg, test_size=0.2, stratify=y_ecg, random_state=42
    )

    # Load PPG — resample to match ECG window length
    X_ppg, y_ppg = load_bidmc(window_seconds=0.5)   # shorter window to match beat length
    # Use same labels as ECG (aligned by index, or just use ECG labels)
    # Simple approach: use minimum N
    N = min(len(y_tr), len(X_ppg))
    rng = np.random.default_rng(42)
    ecg_idx = rng.choice(len(y_tr), N, replace=False)
    ppg_idx = rng.choice(len(X_ppg), N, replace=False)

    X_ecg_use = X_ecg_tr[ecg_idx[:int(N*0.8)]]
    y_use_tr = y_tr[ecg_idx[:int(N*0.8)]]
    X_ppg_use_tr = X_ppg[ppg_idx[:int(N*0.8)]]

    X_ecg_use_v = X_ecg_val[ecg_idx[int(N*0.8):N] % len(X_ecg_val)]
    y_use_val = y_val[ecg_idx[int(N*0.8):N] % len(y_val)]
    X_ppg_use_v = X_ppg[ppg_idx[int(N*0.8):N]]

    # Align window sizes
    win_ecg = X_ecg_use.shape[-1]
    win_ppg = X_ppg_use_tr.shape[-1]

    model = ASCLEPIUSPulse(
        modalities=["ecg", "ppg"],
        n_classes=2,
        d_model=128, n_heads=8, n_cma_layers=3,
        in_channels_override={"ecg": 1, "ppg": 1},
    )

    result = fit_fusion(
        model,
        {"ecg": X_ecg_use, "ppg": X_ppg_use_tr},
        y_use_tr,
        {"ecg": X_ecg_use_v, "ppg": X_ppg_use_v},
        y_use_val,
        epochs=50, batch_size=32, lr=5e-4, seed=42,
        device=DEVICE,
    )
    f1 = result["metrics"].get("f1_macro", 0)
    auc = result["metrics"].get("auc", 0)
    console.print(f"  [green]ASCLEPIUS-PULSE (ECG+PPG)[/green] f1={f1:.4f} auc={auc:.4f}")
    save_results(ROOT / "results" / "fusion" / "ecg_ppg_pulse_real.json", result)
    return result


# ── Federated on MIT-BIH ──────────────────────────────────────────────────────

def run_federated_ecg():
    console.print(Panel("[bold cyan]Module 7 — Federated Learning on MIT-BIH ECG[/bold cyan]"))
    from data.loaders import load_mitbih, train_val_test_split
    from asclepius.module1_per_signal.models import build_model
    from asclepius.module7_federated.federated import (
        FederatedClient, FederatedServer, simulate_hospital_split,
    )

    X, y = load_mitbih(balance=True)
    y_bin = (y > 0).astype(np.int64)  # binary: normal vs arrhythmia
    X_tr, y_tr, X_val, y_val, _, _ = train_val_test_split(X, y_bin)

    hospital_data = simulate_hospital_split(X_tr, y_tr, n_hospitals=5, heterogeneity=0.4)
    clients = []
    for i, (Xh, yh) in enumerate(hospital_data):
        vs = max(1, int(0.2 * len(yh)))
        clients.append(FederatedClient(
            i, Xh[vs:], yh[vs:], Xh[:vs], yh[:vs],
            local_epochs=3, batch_size=64,
            device=DEVICE,
        ))

    global_model = build_model("cnn1d", in_channels=1, n_classes=2, hidden=128)
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        rounds=30,
        dp_enabled=True,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )
    result = server.run()
    f1 = result["final_metrics"].get("f1_macro", (0, 0))[0]
    acc = result["final_metrics"].get("accuracy", (0, 0))[0]
    console.print(f"  [green]FedAvg+DP[/green] acc={acc:.4f} f1={f1:.4f}")
    save_results(ROOT / "results" / "federated" / "ecg_mitbih_fedavg_dp.json", result)
    return result


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: dict):
    table = Table(title="ASCLEPIUS Full Results Summary", show_header=True,
                  header_style="bold magenta")
    table.add_column("Module / Dataset", style="bold")
    table.add_column("Method")
    table.add_column("F1-Macro", justify="right")
    table.add_column("AUC", justify="right")

    if "ecg" in results:
        for arch, m in results["ecg"].items():
            if arch == "baselines":
                for bname, bm in m.items():
                    f1 = bm.get("f1_macro", (0, 0))[0]
                    table.add_row("ECG (MIT-BIH)", bname, f"{f1:.4f}", "-")
            else:
                f1 = m.get("f1_macro", (0, 0))[0]
                auc = m.get("auc", (0, 0))[0]
                table.add_row("ECG (MIT-BIH)", arch, f"{f1:.4f}", f"{auc:.4f}" if auc else "-")

    if "ppg" in results:
        for arch, m in results["ppg"].items():
            f1 = m.get("f1_macro", (0, 0))[0]
            auc = m.get("auc", (0, 0))[0]
            table.add_row("PPG (BIDMC)", arch, f"{f1:.4f}", f"{auc:.4f}" if auc else "-")

    if "anomaly" in results:
        m = results["anomaly"]
        table.add_row("ECG Anomaly", "Training-Free", f"{m.get('f1_macro', 0):.4f}", "-")

    if "fusion" in results:
        m = results["fusion"]["metrics"]
        f1 = m.get("f1_macro", 0)
        auc = m.get("auc", 0)
        table.add_row("ECG+PPG Fusion", "ASCLEPIUS-PULSE", f"{f1:.4f}", f"{auc:.4f}" if auc else "-")

    if "federated" in results:
        m = results["federated"]["final_metrics"]
        f1 = m.get("f1_macro", (0, 0))[0]
        table.add_row("ECG Federated", "FedAvg+DP", f"{f1:.4f}", "-")

    console.print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    console.print(Panel(
        "[bold magenta]ASCLEPIUS — Full Real-Data Experiment Run[/bold magenta]",
        title="ASCLEPIUS",
    ))

    all_results = {}

    try:
        all_results["ecg"] = run_ecg_mitbih()
    except Exception as e:
        logger.error(f"ECG failed: {e}")

    try:
        all_results["ppg"] = run_ppg_bidmc()
    except Exception as e:
        logger.error(f"PPG failed: {e}")

    try:
        all_results["anomaly"] = run_anomaly_ecg()
    except Exception as e:
        logger.error(f"Anomaly failed: {e}")

    try:
        all_results["fusion"] = run_fusion_ecg_ppg()
    except Exception as e:
        logger.error(f"Fusion failed: {e}")

    try:
        all_results["federated"] = run_federated_ecg()
    except Exception as e:
        logger.error(f"Federated failed: {e}")

    print_summary(all_results)
    save_results(ROOT / "results" / "full_real_results.json", all_results)
    console.print(Panel("[bold green]Done! Results saved to results/[/bold green]"))


if __name__ == "__main__":
    main()
