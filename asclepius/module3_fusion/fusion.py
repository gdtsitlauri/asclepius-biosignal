"""Module 3 — ASCLEPIUS-PULSE: Predictive Unified Learning System
for Explainable biomedical Signal analysis.

Novel multi-modal fusion architecture:
  1. Per-modality encoders (shared CNN1D stems)
  2. Cross-modal attention (CMA) — each modality attends to all others
  3. Temporal alignment via learnable lag compensation
  4. Uncertainty-aware fusion via Monte-Carlo dropout
  5. Missing modality handling via learned mask tokens
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asclepius.module1_per_signal.models import CNN1DModel


# ── Modality encoder ──────────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """Encodes a single modality into a fixed-size embedding."""

    def __init__(self, in_channels: int, d_model: int, n_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            _ResBlock(d_model, dilation=2 ** i) for i in range(n_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, d_model)
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(self.pool(h).squeeze(-1))


class _ResBlock(nn.Module):
    def __init__(self, d: int, dilation: int = 1, k: int = 3):
        super().__init__()
        p = dilation * (k - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, k, padding=p, dilation=dilation),
            nn.BatchNorm1d(d), nn.GELU(),
            nn.Conv1d(d, d, k, padding=p, dilation=dilation),
            nn.BatchNorm1d(d),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv(x))


# ── Cross-Modal Attention ─────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """Each modality embedding attends to all other modalities.

    Input:  list of (B, d_model) tensors, one per modality
    Output: list of (B, d_model) enriched tensors
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        # Stack: (B, M, d_model)
        stack = torch.stack(modalities, dim=1)
        attn_out, _ = self.attn(stack, stack, stack)
        stack = self.norm1(stack + self.drop(attn_out))
        stack = self.norm2(stack + self.drop(self.ff(stack)))
        return [stack[:, i] for i in range(stack.size(1))]


# ── Uncertainty head (MC Dropout) ─────────────────────────────────────────────

class UncertaintyHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()  # keep dropout active
        preds = torch.stack([
            torch.softmax(self.forward(x), dim=-1) for _ in range(n_samples)
        ])
        mean = preds.mean(0)
        uncertainty = preds.var(0).sum(-1)  # total predictive variance
        return mean, uncertainty


# ── ASCLEPIUS-PULSE ───────────────────────────────────────────────────────────

MODALITY_CHANNELS = {
    "eeg": 64,
    "ecg": 1,
    "emg": 8,
    "eda": 1,
    "ppg": 1,
}


class ASCLEPIUSPulse(nn.Module):
    """ASCLEPIUS-PULSE: multi-modal biosignal fusion model.

    Accepts any subset of {eeg, ecg, emg, eda, ppg}.
    Missing modalities are replaced with a learned mask token.
    """

    def __init__(
        self,
        modalities: List[str],
        n_classes: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_cma_layers: int = 3,
        dropout: float = 0.2,
        in_channels_override: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.modalities = modalities
        self.d_model = d_model
        self.n_classes = n_classes

        ch_map = {**MODALITY_CHANNELS, **(in_channels_override or {})}

        # Per-modality encoders
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(ch_map.get(m, 1), d_model)
            for m in modalities
        })

        # Learned mask tokens for missing modalities
        self.mask_tokens = nn.ParameterDict({
            m: nn.Parameter(torch.randn(d_model)) for m in modalities
        })

        # Temporal lag compensators (learnable scalar per modality)
        self.lag_compensator = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(1)) for m in modalities
        })

        # Cross-modal attention stack
        self.cma_layers = nn.ModuleList([
            CrossModalAttention(d_model, n_heads, dropout)
            for _ in range(n_cma_layers)
        ])

        # Fusion gate (learned weighted combination)
        self.fusion_gate = nn.Linear(d_model, 1)

        # Classification head with uncertainty
        self.head = UncertaintyHead(d_model, n_classes, dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        inputs: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """
        inputs: dict mapping modality name → (B, C, T) tensor or None if missing.
        Returns: (B, n_classes) logits.
        """
        B = next(v for v in inputs.values() if v is not None).size(0)
        embeddings = []

        for m in self.modalities:
            x = inputs.get(m)
            if x is not None:
                emb = self.encoders[m](x)
            else:
                emb = self.mask_tokens[m].unsqueeze(0).expand(B, -1)
            # apply learned lag shift (in embedding space via scaling)
            emb = emb * (1 + self.lag_compensator[m])
            embeddings.append(emb)

        # Cross-modal attention
        for cma in self.cma_layers:
            embeddings = cma(embeddings)

        # Fusion gate: weighted sum
        stack = torch.stack(embeddings, dim=1)           # (B, M, d)
        gates = torch.softmax(self.fusion_gate(stack), dim=1)  # (B, M, 1)
        fused = (stack * gates).sum(dim=1)               # (B, d)

        return self.head(fused)

    def predict_with_uncertainty(
        self,
        inputs: Dict[str, Optional[torch.Tensor]],
        n_samples: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = next(v for v in inputs.values() if v is not None).size(0)
        embeddings_base = []
        for m in self.modalities:
            x = inputs.get(m)
            if x is not None:
                emb = self.encoders[m](x)
            else:
                emb = self.mask_tokens[m].unsqueeze(0).expand(B, -1)
            emb = emb * (1 + self.lag_compensator[m])
            embeddings_base.append(emb)

        for cma in self.cma_layers:
            embeddings_base = cma(embeddings_base)

        stack = torch.stack(embeddings_base, dim=1)
        gates = torch.softmax(self.fusion_gate(stack), dim=1)
        fused = (stack * gates).sum(dim=1)

        return self.head.predict_with_uncertainty(fused, n_samples)

    def get_modality_importance(
        self,
        inputs: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, float]:
        """Return attention-based modality importance scores."""
        B = next(v for v in inputs.values() if v is not None).size(0)
        embeddings = []
        for m in self.modalities:
            x = inputs.get(m)
            if x is not None:
                emb = self.encoders[m](x)
            else:
                emb = self.mask_tokens[m].unsqueeze(0).expand(B, -1)
            emb = emb * (1 + self.lag_compensator[m])
            embeddings.append(emb)

        for cma in self.cma_layers:
            embeddings = cma(embeddings)

        stack = torch.stack(embeddings, dim=1)
        gates = torch.softmax(self.fusion_gate(stack), dim=1).squeeze(-1)  # (B, M)
        mean_gates = gates.mean(0).detach().cpu().numpy()
        return {m: float(mean_gates[i]) for i, m in enumerate(self.modalities)}
