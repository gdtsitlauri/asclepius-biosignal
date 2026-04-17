#!/usr/bin/env python3
"""Dataset downloader for ASCLEPIUS.

Downloads and prepares all supported biomedical signal datasets.
Run: python data/download_datasets.py --dataset all
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from loguru import logger
from tqdm import tqdm

DATA_DIR = Path(__file__).parent

DATASETS = {
    # ── EEG ───────────────────────────────────────────────────────────────────
    "temple_eeg": {
        "description": "Temple University EEG Corpus (requires registration)",
        "url": "https://isip.piconepress.com/projects/tuh_eeg/",
        "manual": True,
        "instructions": (
            "1. Register at https://isip.piconepress.com/projects/tuh_eeg/\n"
            "2. Download TUEG corpus via rsync (instructions provided upon registration)\n"
            "3. Place data in data/eeg/temple_eeg/"
        ),
    },
    "physionet_eeg": {
        "description": "PhysioNet EEG Motor Movement/Imagery",
        "url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "wfdb_record": "eegmmidb",
        "manual": False,
    },
    # ── ECG ───────────────────────────────────────────────────────────────────
    "mitbih": {
        "description": "MIT-BIH Arrhythmia Database",
        "url": "https://physionet.org/content/mitdb/1.0.0/",
        "wfdb_record": "mitdb",
        "manual": False,
    },
    "physionet_af": {
        "description": "PhysioNet Challenge 2017 — AF Classification",
        "url": "https://physionet.org/content/challenge-2017/1.0.0/",
        "manual": False,
        "wfdb_record": "challenge-2017",
    },
    # ── EMG ───────────────────────────────────────────────────────────────────
    "physionet_emg": {
        "description": "PhysioNet EMG Physical Action Dataset",
        "url": "https://physionet.org/content/emgphysionet/1.0.0/",
        "manual": False,
        "wfdb_record": "emgphysionet",
    },
    "ninapro": {
        "description": "NinaPro EMG Dataset (DB1)",
        "url": "https://ninapro.hevs.ch/",
        "manual": True,
        "instructions": (
            "1. Visit https://ninapro.hevs.ch/\n"
            "2. Register and download DB1\n"
            "3. Place .mat files in data/emg/ninapro/"
        ),
    },
    # ── EDA ───────────────────────────────────────────────────────────────────
    "wesad": {
        "description": "WESAD — Wearable Stress and Affect Detection",
        "url": "https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29",
        "manual": True,
        "instructions": (
            "1. Download from https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx\n"
            "2. Extract to data/eda/wesad/"
        ),
    },
    "deap": {
        "description": "DEAP — Emotion Recognition Dataset",
        "url": "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/",
        "manual": True,
        "instructions": (
            "1. Register at https://www.eecs.qmul.ac.uk/mmv/datasets/deap/\n"
            "2. Download and extract to data/eda/deap/"
        ),
    },
    # ── PPG ───────────────────────────────────────────────────────────────────
    "ppg_dalia": {
        "description": "PPG-DaLiA — HR Estimation During Activities",
        "url": "https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA",
        "manual": True,
        "instructions": (
            "1. Download from UCI ML Repository\n"
            "2. Extract to data/ppg/ppg_dalia/"
        ),
    },
    "bidmc": {
        "description": "BIDMC PPG and Respiration Dataset",
        "url": "https://physionet.org/content/bidmc/1.0.0/",
        "wfdb_record": "bidmc",
        "manual": False,
    },
}


def download_wfdb(record_name: str, output_dir: Path):
    """Download PhysioNet dataset via WFDB."""
    try:
        import wfdb
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {record_name} via WFDB...")
        wfdb.dl_database(record_name, str(output_dir))
        logger.success(f"Downloaded {record_name} → {output_dir}")
    except ImportError:
        logger.error("wfdb not installed. pip install wfdb")
    except Exception as e:
        logger.error(f"Failed to download {record_name}: {e}")


def print_manual_instructions(name: str, info: dict):
    logger.warning(f"\n{'='*60}")
    logger.warning(f"MANUAL DOWNLOAD REQUIRED: {name}")
    logger.warning(f"Description: {info['description']}")
    logger.warning(f"URL: {info['url']}")
    logger.warning("Instructions:")
    for line in info["instructions"].split("\n"):
        logger.warning(f"  {line}")
    logger.warning("=" * 60)


def download_dataset(name: str, info: dict):
    if info.get("manual", False):
        print_manual_instructions(name, info)
        return

    signal_type = "eeg" if "eeg" in name else \
                  "ecg" if any(k in name for k in ["mitbih", "af"]) else \
                  "emg" if "emg" in name else \
                  "ppg" if "ppg" in name or "bidmc" in name else "misc"
    output_dir = DATA_DIR / signal_type / name

    if info.get("wfdb_record"):
        download_wfdb(info["wfdb_record"], output_dir)


def main():
    parser = argparse.ArgumentParser(description="ASCLEPIUS Dataset Downloader")
    parser.add_argument(
        "--dataset", default="all",
        choices=["all"] + list(DATASETS.keys()),
        help="Dataset to download",
    )
    args = parser.parse_args()

    logger.info("ASCLEPIUS Dataset Downloader")
    logger.info(f"Data directory: {DATA_DIR}")

    targets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for name, info in targets.items():
        logger.info(f"\nProcessing: {name} — {info['description']}")
        download_dataset(name, info)

    logger.info("\nDownload complete. Check data/ directory.")
    logger.info("For manual downloads, follow the printed instructions above.")


if __name__ == "__main__":
    main()
