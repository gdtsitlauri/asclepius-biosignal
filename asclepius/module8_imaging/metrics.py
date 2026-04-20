"""Segmentation metrics: Dice, IoU, pixel accuracy, precision, recall."""
from __future__ import annotations

import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Binary Dice coefficient.

    Parameters
    ----------
    pred: predicted binary mask (B, 1, H, W) or (B, H, W), float/bool
    target: ground-truth binary mask, same shape
    """
    pred = (pred > 0.5).float().view(-1)
    target = target.float().view(-1)
    intersection = (pred * target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Intersection over Union (Jaccard index)."""
    pred = (pred > 0.5).float().view(-1)
    target = target.float().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of correctly classified pixels."""
    pred = (pred > 0.5).float().view(-1)
    target = target.float().view(-1)
    return float((pred == target).float().mean())


def precision_recall(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> tuple[float, float]:
    """Binary precision and recall."""
    pred_b = (pred > 0.5).float().view(-1)
    target_b = target.float().view(-1)
    tp = (pred_b * target_b).sum()
    fp = (pred_b * (1 - target_b)).sum()
    fn = ((1 - pred_b) * target_b).sum()
    precision = float((tp + smooth) / (tp + fp + smooth))
    recall = float((tp + smooth) / (tp + fn + smooth))
    return precision, recall


def compute_all_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """Compute Dice, IoU, accuracy, precision, recall in one call."""
    prec, rec = precision_recall(pred, target)
    dice = dice_score(pred, target)
    return {
        "dice": dice,
        "iou": iou_score(pred, target),
        "pixel_accuracy": pixel_accuracy(pred, target),
        "precision": prec,
        "recall": rec,
        "f1": dice,
    }
