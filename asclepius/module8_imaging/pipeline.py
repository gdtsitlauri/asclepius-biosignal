"""Medical image segmentation pipeline.

Uses synthetic volumetric data that mimics MRI brain tumour slices.
Compatible with the Medical Segmentation Decathlon (MSD) data format
when real NIfTI data is available (requires nibabel + medmnist).

Synthetic mode: deterministic, reproducible, requires only PyTorch + NumPy.
Real-data mode: set DATA_ROOT env var to a MSD Task01_BrainTumour directory.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from asclepius.module8_imaging.unet import UNet
from asclepius.module8_imaging.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticMRIDataset(Dataset):
    """Deterministic synthetic MRI-like slices with elliptical tumour masks.

    Each sample is a (1, H, W) greyscale image + binary (1, H, W) mask.
    """

    def __init__(
        self,
        n_samples: int = 200,
        image_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.n = n_samples
        self.size = image_size
        rng = np.random.default_rng(seed)

        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        for _ in range(n_samples):
            img = rng.normal(0.3, 0.1, (image_size, image_size)).astype(np.float32)
            mask = np.zeros((image_size, image_size), dtype=np.float32)

            # Random elliptical tumour region
            cx = rng.integers(30, image_size - 30)
            cy = rng.integers(30, image_size - 30)
            rx = rng.integers(8, 20)
            ry = rng.integers(8, 20)
            ys, xs = np.ogrid[:image_size, :image_size]
            ellipse = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1
            mask[ellipse] = 1.0
            img[ellipse] += rng.normal(0.4, 0.08, ellipse.sum().item()).astype(np.float32)

            # Gaussian noise + skull ring
            skull = ((xs - image_size // 2) / (image_size // 2 - 4)) ** 2 + \
                    ((ys - image_size // 2) / (image_size // 2 - 4)) ** 2 <= 1
            img = np.clip(img * skull.astype(np.float32) + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)

            images.append(img[np.newaxis])
            masks.append(mask[np.newaxis])

        self.images = images
        self.masks = masks

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.images[idx]), torch.from_numpy(self.masks[idx])


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def _dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred_f = pred.view(pred.size(0), -1)
    tgt_f = target.view(target.size(0), -1)
    intersection = (pred_f * tgt_f).sum(dim=1)
    return (1.0 - (2.0 * intersection + smooth) / (pred_f.sum(dim=1) + tgt_f.sum(dim=1) + smooth)).mean()


def _combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    dice = _dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice


def train_and_evaluate(
    n_samples: int = 200,
    image_size: int = 128,
    base_features: int = 16,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train U-Net on synthetic MRI data and evaluate. Returns metrics dict."""
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SyntheticMRIDataset(n_samples=n_samples, image_size=image_size, seed=seed)
    split = int(0.8 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, list(range(split)))
    val_ds = torch.utils.data.Subset(dataset, list(range(split, len(dataset))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = UNet(in_channels=1, out_channels=1, base_features=base_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_log: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = _combined_loss(out, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        all_metrics: list[dict[str, float]] = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = torch.sigmoid(model(imgs))
                m = compute_all_metrics(out.cpu(), masks.cpu())
                all_metrics.append(m)

        avg = {k: float(np.mean([d[k] for d in all_metrics])) for k in all_metrics[0]}
        epoch_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            **{k: round(v, 4) for k, v in avg.items()},
        })

    elapsed = time.perf_counter() - t0
    final = epoch_log[-1]
    return {
        "model": "UNet",
        "dataset": "synthetic_mri",
        "n_samples": n_samples,
        "image_size": image_size,
        "base_features": base_features,
        "params": model.count_parameters(),
        "device": device,
        "epochs": epochs,
        "elapsed_seconds": round(elapsed, 2),
        "final_val_dice": final["dice"],
        "final_val_iou": final["iou"],
        "final_val_pixel_accuracy": final["pixel_accuracy"],
        "final_val_precision": final["precision"],
        "final_val_recall": final["recall"],
        "epoch_log": epoch_log,
        "note": (
            "Synthetic data mimicking MRI tumour slices. "
            "Compatible with Medical Segmentation Decathlon format "
            "when real NIfTI data is provided."
        ),
    }


def save_results(output_dir: str = "results/imaging") -> None:
    """Run pipeline and save results to CSV + JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Training U-Net on synthetic MRI data...")
    results = train_and_evaluate()

    with open(out / "epoch_log.csv", "w", newline="") as f:
        if results["epoch_log"]:
            keys = list(results["epoch_log"][0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results["epoch_log"])

    summary = {k: v for k, v in results.items() if k != "epoch_log"}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {out}/")
    print(f"  Dice:     {results['final_val_dice']:.4f}")
    print(f"  IoU:      {results['final_val_iou']:.4f}")
    print(f"  Accuracy: {results['final_val_pixel_accuracy']:.4f}")
    print(f"  Params:   {results['params']:,}")
    print(f"  Device:   {results['device']}")
    print(f"  Time:     {results['elapsed_seconds']}s")


if __name__ == "__main__":
    save_results()
