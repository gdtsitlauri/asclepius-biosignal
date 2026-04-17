"""Training utilities for ASCLEPIUS-PULSE fusion model."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from asclepius.module3_fusion.fusion import ASCLEPIUSPulse
from asclepius.utils import compute_metrics, set_seed


class MultiModalDataset(Dataset):
    """Dataset for multi-modal biosignal windows.

    modality_data: dict mapping modality name → (N, C, T) ndarray (or None)
    labels: (N,) ndarray
    missing_prob: randomly drop modalities during training for robustness
    """

    def __init__(
        self,
        modality_data: Dict[str, Optional[np.ndarray]],
        labels: np.ndarray,
        missing_prob: float = 0.2,
        training: bool = True,
    ):
        self.modality_data = {
            k: v for k, v in modality_data.items() if v is not None
        }
        self.labels = labels
        self.missing_prob = missing_prob
        self.training = training
        self.modalities = list(self.modality_data.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        for m in self.modalities:
            data = self.modality_data[m][idx]
            if self.training and np.random.rand() < self.missing_prob:
                sample[m] = None
            else:
                sample[m] = torch.from_numpy(data.astype(np.float32))
        label = int(self.labels[idx])
        return sample, label


def collate_multimodal(batch):
    samples, labels = zip(*batch)
    modalities = [k for k in samples[0]]
    batched = {}
    for m in modalities:
        tensors = [s[m] for s in samples]
        valid = [t for t in tensors if t is not None]
        if not valid:
            batched[m] = None
        else:
            ref = valid[0]
            batched[m] = torch.stack([
                t if t is not None else torch.zeros_like(ref) for t in tensors
            ])
    return batched, torch.tensor(labels, dtype=torch.long)


def fit_fusion(
    model: ASCLEPIUSPulse,
    train_data: Dict[str, Optional[np.ndarray]],
    train_labels: np.ndarray,
    val_data: Dict[str, Optional[np.ndarray]],
    val_labels: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict:
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    train_ds = MultiModalDataset(train_data, train_labels, missing_prob=0.2, training=True)
    val_ds = MultiModalDataset(val_data, val_labels, missing_prob=0.0, training=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_multimodal, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        collate_fn=collate_multimodal, num_workers=0,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    bar = tqdm(range(1, epochs + 1), desc="Fusion Training", leave=False)
    for epoch in bar:
        # Train
        model.train()
        total_loss = 0.0
        for batch, y in train_loader:
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
        tr_loss = total_loss / len(train_ds)

        # Val
        model.eval()
        val_loss = 0.0
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch, y in val_loader:
                batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
                y = y.to(device)
                logits = model(batch)
                val_loss += criterion(logits, y).item() * len(y)
                all_pred.append(logits.argmax(-1).cpu().numpy())
                all_true.append(y.cpu().numpy())
        val_loss /= len(val_ds)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_true)
        f1 = compute_metrics(y_true, y_pred).get("f1_macro", 0)
        bar.set_postfix(tr=f"{tr_loss:.4f}", val=f"{val_loss:.4f}", f1=f"{f1:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch, y in val_loader:
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            logits = model(batch)
            all_pred.append(logits.argmax(-1).cpu().numpy())
            all_true.append(y.numpy())
    metrics = compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    return {"history": history, "metrics": metrics}
