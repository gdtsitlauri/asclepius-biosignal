"""Module 7 — Federated Learning with Differential Privacy.

Simulates a multi-hospital federated training environment.
- FedAvg aggregation
- Differential Privacy via Gaussian noise mechanism
- GDPR-compliant: no raw patient data sharing
- Comparison vs centralized training
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from asclepius.utils import aggregate_metrics, compute_metrics, set_seed


# ── Differential Privacy ──────────────────────────────────────────────────────

class DPGaussianMechanism:
    """Add Gaussian noise to model gradients for differential privacy."""

    def __init__(self, noise_multiplier: float = 1.1, max_grad_norm: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def clip_and_noise(self, model: nn.Module) -> nn.Module:
        """Clip gradients and add calibrated Gaussian noise in-place."""
        # Clip
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        # Add Gaussian noise
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    0, self.noise_multiplier * self.max_grad_norm,
                    size=p.grad.shape, device=p.grad.device,
                )
                p.grad.data.add_(noise)
        return model


# ── FedAvg ────────────────────────────────────────────────────────────────────

def fedavg_aggregate(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_sizes: List[int],
) -> nn.Module:
    """Weighted FedAvg: weight by dataset size."""
    total = sum(client_sizes)
    global_state = global_model.state_dict()

    for key in global_state:
        weighted = torch.zeros_like(global_state[key].cpu(), dtype=torch.float32)
        for model, size in zip(client_models, client_sizes):
            weighted += model.state_dict()[key].cpu().float() * (size / total)
        global_state[key] = weighted

    global_model.load_state_dict(global_state)
    return global_model


# ── Client ────────────────────────────────────────────────────────────────────

@dataclass
class FederatedClient:
    """Simulates a hospital client in federated training."""

    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    local_epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    dp: Optional[DPGaussianMechanism] = None
    device: Optional[torch.device] = None

    def local_train(self, global_model: nn.Module) -> Tuple[nn.Module, int]:
        """Train local copy and return updated model + dataset size."""
        from torch.utils.data import DataLoader, TensorDataset
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = copy.deepcopy(global_model).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.from_numpy(self.X_train.astype(np.float32))
        y_t = torch.from_numpy(self.y_train.astype(np.int64))
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )

        model.train()
        for _ in range(self.local_epochs):
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                if self.dp is not None:
                    self.dp.clip_and_noise(model)
                optimizer.step()

        return model, len(self.X_train)

    def evaluate(self, model: nn.Module) -> Dict:
        from torch.utils.data import DataLoader, TensorDataset
        device = self.device or torch.device("cpu")
        model.eval().to(device)
        X_t = torch.from_numpy(self.X_val.astype(np.float32))
        y_t = torch.from_numpy(self.y_val.astype(np.int64))
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64)
        preds, trues = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                preds.append(model(X_b.to(device)).argmax(-1).cpu().numpy())
                trues.append(y_b.numpy())
        return compute_metrics(np.concatenate(trues), np.concatenate(preds))


# ── Federated Server ──────────────────────────────────────────────────────────

@dataclass
class FederatedServer:
    """Orchestrates federated training across hospital clients."""

    global_model: nn.Module
    clients: List[FederatedClient]
    rounds: int = 50
    fraction_fit: float = 0.8
    dp_enabled: bool = True
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    seed: int = 42

    def run(self) -> Dict:
        set_seed(self.seed)
        dp = DPGaussianMechanism(self.noise_multiplier, self.max_grad_norm) if self.dp_enabled else None
        if dp is not None:
            for c in self.clients:
                c.dp = dp

        round_metrics = []
        n_clients = max(1, int(len(self.clients) * self.fraction_fit))

        logger.info(
            f"[FedServer] Starting {self.rounds} rounds | "
            f"{n_clients}/{len(self.clients)} clients/round | "
            f"DP={'ON' if dp else 'OFF'}"
        )

        for rnd in range(1, self.rounds + 1):
            rng = np.random.default_rng(self.seed + rnd)
            selected = rng.choice(self.clients, size=n_clients, replace=False).tolist()

            client_models, client_sizes = [], []
            for client in selected:
                m, n = client.local_train(self.global_model)
                client_models.append(m)
                client_sizes.append(n)

            self.global_model = fedavg_aggregate(self.global_model, client_models, client_sizes)

            # Evaluate on all client validation sets
            all_metrics = [c.evaluate(self.global_model) for c in self.clients]
            agg = aggregate_metrics(all_metrics)
            round_metrics.append({k: v[0] for k, v in agg.items()})

            if rnd % 10 == 0 or rnd == 1:
                f1 = agg.get("f1_macro", (0, 0))[0]
                acc = agg.get("accuracy", (0, 0))[0]
                logger.info(f"  Round {rnd:3d}/{self.rounds} | acc={acc:.4f} f1={f1:.4f}")

        final = aggregate_metrics([c.evaluate(self.global_model) for c in self.clients])
        return {
            "final_metrics": final,
            "round_metrics": round_metrics,
            "n_rounds": self.rounds,
            "dp_enabled": self.dp_enabled,
            "n_clients": len(self.clients),
        }


# ── Simulation helpers ────────────────────────────────────────────────────────

def simulate_hospital_split(
    X: np.ndarray,
    y: np.ndarray,
    n_hospitals: int = 5,
    heterogeneity: float = 0.5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split dataset into hospital shards with configurable non-IIDness."""
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(y))
    hospital_data = []
    indices = np.arange(len(y))

    for h in range(n_hospitals):
        # Non-IID: each hospital sees some classes more frequently
        class_probs = rng.dirichlet(
            np.ones(n_classes) * (1 - heterogeneity) + rng.random(n_classes) * heterogeneity
        )
        hospital_idx = []
        target_size = len(y) // n_hospitals
        for _ in range(target_size):
            chosen_class = rng.choice(n_classes, p=class_probs)
            class_idx = indices[y == chosen_class]
            if len(class_idx) > 0:
                hospital_idx.append(rng.choice(class_idx))
        hospital_idx = np.array(hospital_idx)
        hospital_data.append((X[hospital_idx], y[hospital_idx]))

    return hospital_data
