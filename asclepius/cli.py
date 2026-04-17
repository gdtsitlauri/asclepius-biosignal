"""ASCLEPIUS CLI entry point."""
from __future__ import annotations
import sys
import click
from pathlib import Path


@click.group()
def main():
    """ASCLEPIUS — Biomedical Signal AI Framework."""
    pass


@main.command()
@click.option("--module", default=0, type=int, help="Module to run (0=all, 1-7=specific)")
@click.option("--quick", is_flag=True, help="Quick run with small synthetic data")
def run(module, quick):
    """Run ASCLEPIUS experiments."""
    import subprocess, sys
    root = Path(__file__).parent.parent
    cmd = [sys.executable, str(root / "experiments" / "run_all.py")]
    if module:
        cmd += ["--module", str(module)]
    if quick:
        cmd.append("--quick")
    subprocess.run(cmd)


@main.command()
def dashboard():
    """Launch the real-time Streamlit dashboard."""
    import subprocess, sys
    root = Path(__file__).parent.parent
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(root / "dashboard" / "app.py")])


@main.command()
@click.option("--dataset", default="all", help="Dataset to download")
def download(dataset):
    """Download biomedical signal datasets."""
    import subprocess, sys
    root = Path(__file__).parent.parent
    subprocess.run([sys.executable, str(root / "data" / "download_datasets.py"),
                    "--dataset", dataset])


if __name__ == "__main__":
    main()
