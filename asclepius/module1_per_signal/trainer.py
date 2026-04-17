"""Training and evaluation loop for deep learning models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from asclepius.utils import compute_metrics, set_seed


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    def to_tensors(X, y):
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))
        return TensorDataset(X_t, y_t)

    train_ds = to_tensors(X_train, y_train)
    val_ds = to_tensors(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    all_pred, all_true, all_prob = [], [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        prob = torch.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
        all_prob.append(prob.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    binary_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else None
    return total_loss / len(loader.dataset), compute_metrics(y_true, y_pred, binary_prob)


def fit(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict:
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "val_metrics": []}
    bar = tqdm(range(1, epochs + 1), desc="Training", leave=False)
    for epoch in bar:
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)
        bar.set_postfix(
            tr=f"{tr_loss:.4f}",
            val=f"{val_loss:.4f}",
            f1=f"{val_metrics.get('f1_macro', 0):.4f}",
        )
        if stopper.step(val_loss, model):
            break

    stopper.restore(model)
    _, final_metrics = eval_epoch(model, val_loader, criterion, device)
    return {"history": history, "metrics": final_metrics}


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_pred, all_prob = [], []
    for (X_b,) in loader:
        logits = model(X_b.to(device))
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        all_prob.append(prob)
        all_pred.append(prob.argmax(axis=-1))
    return np.concatenate(all_pred), np.concatenate(all_prob)
