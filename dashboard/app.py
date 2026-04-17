"""ASCLEPIUS Real-Time Dashboard — Streamlit application.

Run: streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="ASCLEPIUS — Biomedical Signal AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 ASCLEPIUS")
    st.caption("Adaptive Signal Classification and Learning Engine")
    st.divider()

    mode = st.selectbox(
        "Mode",
        ["Real-Time Monitor", "Offline Analysis", "Multi-Modal Fusion", "Explainability"],
    )

    signal_type = st.selectbox(
        "Signal Type",
        ["EEG", "ECG", "EMG", "EDA", "PPG"],
    )

    source = st.selectbox(
        "Data Source",
        ["Synthetic (Demo)", "OpenBCI (BrainFlow)", "File Upload"],
    )

    if source == "File Upload":
        uploaded = st.file_uploader("Upload signal file (.npy, .csv)", type=["npy", "csv"])

    st.divider()
    st.subheader("Detection Settings")
    threshold = st.slider("Anomaly Threshold (σ)", 1.0, 5.0, 3.0, 0.1)
    window_s = st.slider("Window (seconds)", 1, 30, 4)

    st.divider()
    st.subheader("About")
    st.info(
        "**ASCLEPIUS-PULSE**\n\n"
        "Novel multi-modal biosignal fusion combining EEG+ECG+EMG+EDA+PPG "
        "with cross-modal attention and uncertainty estimation.\n\n"
        "License: MIT | Python 3.11+ | PyTorch"
    )


# ── Main content ──────────────────────────────────────────────────────────────

st.title("🧬 ASCLEPIUS — Biomedical Signal Analysis")

if mode == "Real-Time Monitor":
    _render_realtime(signal_type.lower(), threshold, window_s, source)
elif mode == "Offline Analysis":
    _render_offline(signal_type.lower())
elif mode == "Multi-Modal Fusion":
    _render_fusion()
elif mode == "Explainability":
    _render_explainability(signal_type.lower())


# ── Page renderers ─────────────────────────────────────────────────────────────

def _render_realtime(sig: str, threshold: float, window_s: int, source: str):
    import plotly.graph_objects as go

    st.subheader(f"Real-Time {sig.upper()} Monitoring")

    col1, col2, col3, col4 = st.columns(4)
    score_ph = col1.empty()
    status_ph = col2.empty()
    hr_ph = col3.empty()
    alert_ph = col4.empty()

    chart_ph = st.empty()
    log_ph = st.empty()

    # Initialize synthetic source
    from asclepius.module5_realtime.monitor import SyntheticStreamSource, RealTimeMonitor
    from asclepius.module2_anomaly.detector import AsclepiusAnomalyDetector

    FS_MAP = {"eeg": 256, "ecg": 360, "emg": 2000, "eda": 4, "ppg": 64}
    CH_MAP = {"eeg": 4, "ecg": 1, "emg": 4, "eda": 1, "ppg": 1}
    fs = FS_MAP.get(sig, 256)
    n_ch = CH_MAP.get(sig, 1)

    stream = SyntheticStreamSource(sig, n_ch, float(fs), chunk_size=32)
    detector = AsclepiusAnomalyDetector(sig, float(fs), threshold_sigma=threshold)
    monitor = RealTimeMonitor(
        signal_type=sig, n_channels=n_ch, sampling_rate=float(fs),
        window_seconds=float(window_s), step_seconds=0.5,
        anomaly_detector=detector,
    )

    # Buffer for plotting
    plot_buffer = []
    alert_log = []

    if st.button("▶ Start Monitoring", type="primary"):
        monitor.start(stream)
        stop = st.button("⏹ Stop")

        while not stop:
            data = stream.read()
            if data is not None:
                plot_buffer.extend(data[0].tolist())
                if len(plot_buffer) > fs * 10:
                    plot_buffer = plot_buffer[-fs * 10:]

            alerts = monitor.get_pending_alerts()
            for alert in alerts:
                if alert.is_anomaly:
                    alert_log.append(f"🚨 {time.strftime('%H:%M:%S')} | {alert.explanation}")
                score_ph.metric("Anomaly Score", f"{alert.anomaly_score:.2f}σ",
                                delta=f"{alert.anomaly_score - threshold:+.2f}")
                status = "🟢 Normal" if not alert.is_anomaly else "🔴 ANOMALY"
                status_ph.metric("Status", status)

            if plot_buffer:
                t_axis = np.arange(len(plot_buffer)) / fs
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=t_axis, y=plot_buffer,
                    mode="lines", name=sig.upper(),
                    line=dict(color="#00bcd4", width=1),
                ))
                fig.update_layout(
                    title=f"{sig.upper()} Signal (last {window_s * 2}s)",
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                chart_ph.plotly_chart(fig, use_container_width=True)

            if alert_log:
                log_ph.text_area("Alert Log", "\n".join(alert_log[-10:]), height=150)

            time.sleep(0.1)

        monitor.stop()
        stream.stop()


def _render_offline(sig: str):
    st.subheader(f"Offline {sig.upper()} Analysis")
    import plotly.graph_objects as go
    import plotly.express as px

    st.info("Upload a signal file or use synthetic data to run offline analysis.")

    if st.button("Generate Synthetic Sample"):
        fs_map = {"eeg": 256, "ecg": 360, "emg": 2000, "eda": 4, "ppg": 64}
        ch_map = {"eeg": 4, "ecg": 1, "emg": 4, "eda": 1, "ppg": 1}
        fs = fs_map.get(sig, 256)
        n_ch = ch_map.get(sig, 1)
        win = int(fs * 4)
        t = np.linspace(0, 4, win)
        signal = np.stack([
            np.sin(2 * np.pi * 10 * t) * 15 + np.random.randn(win) * 2
            for _ in range(n_ch)
        ])

        # Plot
        fig = go.Figure()
        for i, ch in enumerate(signal):
            fig.add_trace(go.Scatter(x=t, y=ch, name=f"Ch {i+1}", mode="lines"))
        fig.update_layout(template="plotly_dark", title=f"Synthetic {sig.upper()}", height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Run anomaly detection
        from asclepius.module2_anomaly.detector import AsclepiusAnomalyDetector
        detector = AsclepiusAnomalyDetector(sig, float(fs))
        result = detector.score(signal)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Anomaly Score", f"{result['anomaly_score']:.2f}σ")
            st.metric("Status", "🟢 Normal" if not result["is_anomaly"] else "🔴 ANOMALY")
        with col2:
            st.write("**Feature Deviations:**")
            for feat, z in sorted(result["per_feature_zscore"].items(), key=lambda x: -x[1])[:5]:
                st.progress(min(z / 5, 1.0), text=f"{feat}: {z:.2f}σ")

        # Report
        from asclepius.module6_explainability.explainer import MedicalReportGenerator
        rg = MedicalReportGenerator()
        report = rg.generate(sig, result)
        with st.expander("📋 Medical Report"):
            st.text(report)


def _render_fusion():
    st.subheader("ASCLEPIUS-PULSE — Multi-Modal Fusion")
    import plotly.graph_objects as go

    st.markdown("""
    **ASCLEPIUS-PULSE** combines up to 5 signal modalities simultaneously:
    - Cross-Modal Attention (CMA) between all signal types
    - Uncertainty-aware predictions via MC Dropout
    - Missing modality handling via learned mask tokens
    """)

    selected_mods = st.multiselect(
        "Select Modalities",
        ["EEG", "ECG", "EMG", "EDA", "PPG"],
        default=["EEG", "ECG", "PPG"],
    )

    if st.button("Run Fusion Demo") and selected_mods:
        from asclepius.module3_fusion.fusion import ASCLEPIUSPulse
        import torch

        mods = [m.lower() for m in selected_mods]
        ch_map = {"eeg": 4, "ecg": 1, "emg": 4, "eda": 1, "ppg": 1}
        win_map = {"eeg": 512, "ecg": 256, "emg": 128, "eda": 64, "ppg": 128}

        model = ASCLEPIUSPulse(
            modalities=mods, n_classes=3, d_model=64,
            n_heads=4, n_cma_layers=2,
            in_channels_override={m: ch_map[m] for m in mods},
        )

        inputs = {
            m: torch.randn(1, ch_map[m], win_map[m]) for m in mods
        }

        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        importance = model.get_modality_importance(inputs)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Class Probabilities:**")
            classes = ["Normal", "Anomaly", "Pre-onset"]
            fig = go.Figure(go.Bar(x=classes, y=probs, marker_color=["green", "red", "orange"]))
            fig.update_layout(template="plotly_dark", height=250)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Modality Importance (Fusion Gate):**")
            mods_sorted = sorted(importance.items(), key=lambda x: -x[1])
            fig2 = go.Figure(go.Bar(
                x=[m[0].upper() for m in mods_sorted],
                y=[m[1] for m in mods_sorted],
                marker_color="#00bcd4",
            ))
            fig2.update_layout(template="plotly_dark", height=250)
            st.plotly_chart(fig2, use_container_width=True)


def _render_explainability(sig: str):
    st.subheader(f"Explainability — {sig.upper()}")
    st.markdown("""
    ASCLEPIUS provides multiple explainability methods:
    - **Grad-CAM**: Temporal saliency maps showing which time regions matter
    - **SHAP**: Feature importance for classical ML models
    - **Medical Reports**: Human-readable clinical summaries
    """)

    if st.button("Generate Explanation Demo"):
        import plotly.graph_objects as go

        fs_map = {"eeg": 256, "ecg": 360, "emg": 2000, "eda": 4, "ppg": 64}
        fs = fs_map.get(sig, 256)
        win = int(fs * 4)
        t = np.linspace(0, 4, win)

        # Synthetic CAM
        cam = np.abs(np.sin(2 * np.pi * 0.5 * t) + np.random.randn(win) * 0.2)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        signal = np.sin(2 * np.pi * 10 * t) * 15 + np.random.randn(win) * 2

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=signal, name=sig.upper(),
                                 line=dict(color="#00bcd4")))
        fig.add_trace(go.Scatter(x=t, y=cam * 30, name="Grad-CAM Saliency",
                                 fill="tozeroy", line=dict(color="rgba(255,100,0,0.5)")))
        fig.update_layout(template="plotly_dark", title="Signal + Grad-CAM Saliency", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Red shaded regions show temporal locations most influential "
            "for the model's prediction (Grad-CAM activation)."
        )
