"""Deep learning models for per-signal analysis: CNN1D, LSTM, Transformer."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── CNN1D ─────────────────────────────────────────────────────────────────────

class CNN1DModel(nn.Module):
    """Multi-scale dilated CNN for biomedical time series."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hidden: int = 128,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden, dilation=dilations[i % len(dilations)])
            for i in range(n_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        h = self.pool(h).squeeze(-1)
        return self.head(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.pool(h).squeeze(-1)


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """Bidirectional LSTM with attention for biomedical time series."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden)
        self.lstm = nn.LSTM(
            hidden, hidden, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        h = self.proj(x)
        out, _ = self.lstm(h)
        # attention over time
        scores = self.attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (out * weights).sum(dim=1)
        return self.head(ctx)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        h = self.proj(x)
        out, _ = self.lstm(h)
        scores = self.attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (out * weights).sum(dim=1)


# ── Transformer ───────────────────────────────────────────────────────────────

class TransformerModel(nn.Module):
    """Transformer encoder for biomedical time series classification."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers,
                                              norm=nn.LayerNorm(d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        h = self.pos_enc(self.proj(x))
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(h)
        return self.head(h[:, 0])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        h = self.pos_enc(self.proj(x))
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(h)
        return h[:, 0]


# ── Model factory ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "cnn1d": CNN1DModel,
    "lstm": LSTMModel,
    "transformer": TransformerModel,
}


def build_model(
    arch: str,
    in_channels: int,
    n_classes: int,
    **kwargs,
) -> nn.Module:
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](in_channels, n_classes, **kwargs)
