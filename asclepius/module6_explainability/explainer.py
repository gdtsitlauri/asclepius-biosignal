"""Module 6 — Explainability.

  1. Grad-CAM for temporal signal regions (CNN1D / Transformer)
  2. SHAP values for feature importance (baselines + DL)
  3. Human-readable medical report generation
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM1D:
    """Grad-CAM for 1-D temporal models (CNN1D, any model with conv layers).

    Computes class activation maps over the temporal dimension.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self._activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Return CAM of shape (T,) — same length as input time axis."""
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # activations: (1, C_feat, T) ; gradients: same
        acts = self._activations[0]         # (C_feat, T)
        grads = self._gradients[0]          # (C_feat, T)
        weights = grads.mean(dim=-1)        # (C_feat,)
        cam = (weights[:, None] * acts).sum(0)  # (T,)
        cam = torch.relu(cam).cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

    def batch_generate(
        self,
        X: torch.Tensor,
        class_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        cams = []
        for i, x in enumerate(X):
            ci = class_indices[i] if class_indices else None
            cams.append(self.generate(x.unsqueeze(0), ci))
        return np.stack(cams)


# ── SHAP ──────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """SHAP-based feature importance for classical ML and deep models."""

    def __init__(self, model, background_data: np.ndarray, model_type: str = "tree"):
        """
        model_type: 'tree' (RF/LightGBM), 'deep' (PyTorch), 'kernel' (SVM)
        """
        import shap
        self.model_type = model_type
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "kernel":
            self.explainer = shap.KernelExplainer(
                model.predict_proba, shap.sample(background_data, 50)
            )
        else:
            raise NotImplementedError("Deep SHAP requires custom wrapper — use GradientExplainer.")
        self.shap = shap

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values array (n_samples, n_features) or (n_samples, n_features, n_classes)."""
        sv = self.explainer.shap_values(X)
        return sv

    def top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        if isinstance(shap_values, list):
            # multi-class: take class-1 SHAP
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
        mean_abs = np.abs(sv).mean(0)
        idx = np.argsort(mean_abs)[::-1][:top_k]
        return [(feature_names[i], float(mean_abs[i])) for i in idx]


# ── Medical Report Generator ──────────────────────────────────────────────────

class MedicalReportGenerator:
    """Generates human-readable medical reports from analysis results."""

    DISEASE_DESCRIPTIONS = {
        "epilepsy": "Epilepsy (seizure disorder)",
        "alzheimer": "Alzheimer's Disease",
        "depression": "Major Depressive Disorder",
        "arrhythmia": "Cardiac Arrhythmia",
        "atrial_fibrillation": "Atrial Fibrillation",
        "neuromuscular": "Neuromuscular Disorder",
        "stress": "Stress / Anxiety",
        "burnout": "Burnout Syndrome",
        "normal": "Normal / Healthy",
    }

    RISK_LEVELS = {
        (0.0, 0.3): ("LOW", "No significant abnormalities detected."),
        (0.3, 0.6): ("MODERATE", "Some abnormal patterns detected. Monitor closely."),
        (0.6, 0.8): ("HIGH", "Significant abnormalities present. Medical evaluation recommended."),
        (0.8, 1.0): ("CRITICAL", "Critical abnormalities detected. Urgent medical attention required."),
    }

    def generate(
        self,
        signal_type: str,
        anomaly_result: Dict,
        prediction_result: Optional[Dict] = None,
        modality_importance: Optional[Dict] = None,
        top_shap_features: Optional[List[Tuple[str, float]]] = None,
        patient_id: Optional[str] = None,
    ) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("ASCLEPIUS BIOMEDICAL SIGNAL ANALYSIS REPORT")
        lines.append("=" * 70)
        if patient_id:
            lines.append(f"Patient ID  : {patient_id}")
        lines.append(f"Signal Type : {signal_type.upper()}")

        import datetime
        lines.append(f"Generated   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Anomaly section
        score = anomaly_result.get("anomaly_score", 0.0)
        is_anomaly = anomaly_result.get("is_anomaly", False)
        risk_level, risk_text = self._risk_level(score / 5.0)  # normalize ~3σ → 0.6
        lines.append("── ANOMALY ANALYSIS ──────────────────────────────────────────────")
        lines.append(f"Status      : {'⚠ ANOMALY DETECTED' if is_anomaly else '✓ Normal Signal'}")
        lines.append(f"Risk Level  : {risk_level}")
        lines.append(f"Anomaly σ   : {score:.2f}")
        lines.append(f"Assessment  : {risk_text}")
        lines.append("")

        # Per-feature breakdown
        pf = anomaly_result.get("per_feature_zscore", {})
        if pf:
            lines.append("── FEATURE DEVIATION (σ from healthy baseline) ──────────────────")
            for feat, z in sorted(pf.items(), key=lambda x: -x[1])[:8]:
                bar = "█" * int(min(z, 10)) + "░" * max(0, 10 - int(min(z, 10)))
                lines.append(f"  {feat:<30} {bar}  {z:.2f}σ")
            lines.append("")

        # Prediction section
        if prediction_result:
            lines.append("── DISEASE PREDICTION ───────────────────────────────────────────")
            pred_class = prediction_result.get("predicted_class", "unknown")
            prob = prediction_result.get("probability", 0.0)
            unc = prediction_result.get("uncertainty", None)
            desc = self.DISEASE_DESCRIPTIONS.get(str(pred_class), str(pred_class))
            lines.append(f"Prediction  : {desc}")
            lines.append(f"Confidence  : {prob * 100:.1f}%")
            if unc is not None:
                lines.append(f"Uncertainty : {unc:.4f} (lower=more reliable)")
            lines.append("")

        # Modality importance
        if modality_importance:
            lines.append("── MODALITY CONTRIBUTION (ASCLEPIUS-PULSE) ──────────────────────")
            for mod, imp in sorted(modality_importance.items(), key=lambda x: -x[1]):
                bar = "█" * int(imp * 20) + "░" * max(0, 20 - int(imp * 20))
                lines.append(f"  {mod.upper():<10} {bar}  {imp * 100:.1f}%")
            lines.append("")

        # SHAP features
        if top_shap_features:
            lines.append("── TOP PREDICTIVE FEATURES (SHAP) ───────────────────────────────")
            for feat, val in top_shap_features[:6]:
                lines.append(f"  {feat:<35} impact={val:.4f}")
            lines.append("")

        # Explanation
        expl = anomaly_result.get("explanation", "")
        if expl:
            lines.append("── CLINICAL SUMMARY ──────────────────────────────────────────────")
            lines.append(f"  {expl}")
            lines.append("")

        lines.append("── DISCLAIMER ────────────────────────────────────────────────────")
        lines.append("  This report is generated by an AI system and is intended for")
        lines.append("  research purposes only. Always consult a qualified medical")
        lines.append("  professional for clinical decisions.")
        lines.append("=" * 70)

        return "\n".join(lines)

    def save(self, report: str, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report)

    def _risk_level(self, normalized_score: float) -> Tuple[str, str]:
        normalized_score = max(0.0, min(normalized_score, 0.9999))
        for (lo, hi), (level, text) in self.RISK_LEVELS.items():
            if lo <= normalized_score < hi:
                return level, text
        return "LOW", "No significant abnormalities detected."
