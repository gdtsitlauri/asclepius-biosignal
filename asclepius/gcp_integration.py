"""
Google Cloud Platform integration for ASCLEPIUS biomedical AI framework.

Provides Google Cloud Storage for dataset/model management and
Vertex AI client for cloud-based training job submission.

Credentials
-----------
Set via environment variable:
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json

Or use Application Default Credentials (ADC) on GCE/Cloud Run:
    gcloud auth application-default login

Usage
-----
    from asclepius.gcp_integration import GCPProvider

    gcp = GCPProvider(project="my-gcp-project", bucket="asclepius-data")
    gcp.upload_model("results/unet_best.pt", "models/unet_best.pt")
    gcp.upload_dataset_split("data/mri_train.csv", "datasets/mri_train.csv")
    job_name = gcp.submit_vertex_training(
        display_name="asclepius_unet_run",
        script_path="asclepius/module8_imaging/train_unet.py",
        machine_type="n1-standard-8",
        accelerator="NVIDIA_TESLA_T4",
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from google.cloud import storage as gcs
    _GCS_AVAILABLE = True
except ImportError:
    _GCS_AVAILABLE = False

try:
    from google.cloud import aiplatform
    _VERTEX_AVAILABLE = True
except ImportError:
    _VERTEX_AVAILABLE = False


_DEFAULT_REGION = os.getenv("GCP_REGION", "europe-west1")


def _check_gcs():
    if not _GCS_AVAILABLE:
        raise ImportError(
            "google-cloud-storage not installed.\n"
            "Run: pip install google-cloud-storage"
        )


def _check_vertex():
    if not _VERTEX_AVAILABLE:
        raise ImportError(
            "google-cloud-aiplatform not installed.\n"
            "Run: pip install google-cloud-aiplatform"
        )


class GCPProvider:
    """GCP provider for ASCLEPIUS dataset, model, and training management."""

    def __init__(
        self,
        project: str,
        bucket: str = "asclepius-biomedical",
        region: str = _DEFAULT_REGION,
    ) -> None:
        self.project = project
        self.bucket_name = bucket
        self.region = region
        self._client: Any = None

    def _gcs(self):
        _check_gcs()
        if self._client is None:
            self._client = gcs.Client(project=self.project)
        return self._client

    # ── Google Cloud Storage ───────────────────────────────────────────

    def upload_model(self, local_path: str, gcs_path: str) -> str:
        """Upload a trained model checkpoint to GCS. Returns gs:// URI."""
        bucket = self._gcs().bucket(self.bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        uri = f"gs://{self.bucket_name}/{gcs_path}"
        print(f"[GCP GCS] Uploaded {local_path} → {uri}")
        return uri

    def upload_dataset_split(self, local_path: str, gcs_path: str) -> str:
        """Upload a dataset CSV/NPZ split to GCS."""
        return self.upload_model(local_path, gcs_path)

    def download_artifact(self, gcs_path: str, local_path: str) -> None:
        """Download a GCS object to a local path."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        bucket = self._gcs().bucket(self.bucket_name)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f"[GCP GCS] Downloaded gs://{self.bucket_name}/{gcs_path} → {local_path}")

    def list_artifacts(self, prefix: str = "models/") -> list[str]:
        """List all GCS blobs under a prefix."""
        bucket = self._gcs().bucket(self.bucket_name)
        return [b.name for b in bucket.list_blobs(prefix=prefix)]

    def upload_results(self, results: dict[str, Any], run_id: str) -> str:
        """Serialize a results dict as JSON and upload to GCS."""
        tmp = Path(f"/tmp/asclepius_run_{run_id}.json")
        tmp.write_text(json.dumps(results, indent=2))
        return self.upload_model(str(tmp), f"results/{run_id}.json")

    # ── Vertex AI ─────────────────────────────────────────────────────

    def submit_vertex_training(
        self,
        display_name: str,
        script_path: str,
        machine_type: str = "n1-standard-8",
        accelerator: str = "NVIDIA_TESLA_T4",
        accelerator_count: int = 1,
    ) -> str:
        """Submit a custom training job to Vertex AI. Returns job name."""
        _check_vertex()
        aiplatform.init(project=self.project, location=self.region)

        job = aiplatform.CustomJob.from_local_script(
            display_name=display_name,
            script_path=script_path,
            requirements=["torch>=2.0", "scikit-learn", "numpy", "pandas"],
            machine_type=machine_type,
            accelerator_type=accelerator,
            accelerator_count=accelerator_count,
        )
        job.submit()
        print(f"[GCP Vertex AI] Submitted job: {job.display_name} ({job.resource_name})")
        return job.resource_name

    def list_vertex_jobs(self) -> list[dict]:
        """List recent Vertex AI custom training jobs."""
        _check_vertex()
        aiplatform.init(project=self.project, location=self.region)
        jobs = aiplatform.CustomJob.list()
        return [{"name": j.display_name, "state": str(j.state)} for j in jobs]
