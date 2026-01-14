"""Configuration models for SpotJAX."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


def get_default_project() -> str | None:
    """Get the default GCP project from environment or gcloud config."""
    if project := os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return project
    if project := os.environ.get("GCLOUD_PROJECT"):
        return project
    if project := os.environ.get("CLOUDSDK_CORE_PROJECT"):
        return project
    return None


def generate_job_id() -> str:
    """Generate a timestamp-based job ID."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class TPUConfig(BaseModel):
    """Configuration for TPU provisioning."""

    tpu_type: Annotated[str, Field(description="TPU type (e.g., v4-8, v4-32, v5p-16)")]
    zone: Annotated[str, Field(description="GCP zone (e.g., us-central2-b)")]
    project: Annotated[str, Field(description="GCP project ID")]
    spot: Annotated[bool, Field(default=True, description="Use spot/preemptible TPUs")]

    @field_validator("tpu_type")
    @classmethod
    def validate_tpu_type(cls, v: str) -> str:
        """Validate TPU type format."""
        pattern = r"^v[45]p?-(8|16|32|64|128|256|512|1024|2048|4096)$"
        if not re.match(pattern, v):
            raise ValueError(
                f"Invalid TPU type '{v}'. Expected format like v4-8, v4-32, v5p-16"
            )
        return v

    @field_validator("zone")
    @classmethod
    def validate_zone(cls, v: str) -> str:
        """Validate zone format."""
        pattern = r"^[a-z]+-[a-z]+\d+-[a-z]$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid zone format '{v}'. Expected format like us-central2-b")
        return v

    @property
    def accelerator_type(self) -> str:
        """Get the accelerator type string for GCP API."""
        return self.tpu_type

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the TPU slice."""
        match = re.search(r"-(\d+)$", self.tpu_type)
        if not match:
            return 1
        chips = int(match.group(1))
        # v4 and v5p have 4 chips per node
        return max(1, chips // 4)


class RunConfig(BaseModel):
    """Configuration for a training run."""

    script: Annotated[Path, Field(description="Path to the training script")]
    code_dir: Annotated[Path, Field(description="Directory containing user code to sync")]
    max_retries: Annotated[int, Field(default=5, ge=0, le=100, description="Max restart attempts")]
    stream_worker: Annotated[int, Field(default=0, ge=0, description="Worker index to stream logs from")]

    @field_validator("script", mode="before")
    @classmethod
    def resolve_script(cls, v: str | Path) -> Path:
        """Resolve script path to absolute."""
        return Path(v).resolve()

    @field_validator("code_dir", mode="before")
    @classmethod
    def resolve_code_dir(cls, v: str | Path) -> Path:
        """Resolve code directory to absolute."""
        return Path(v).resolve()

    @model_validator(mode="after")
    def validate_paths(self) -> RunConfig:
        """Validate that script exists and is within code_dir."""
        if not self.code_dir.exists():
            raise ValueError(f"Code directory does not exist: {self.code_dir}")
        if not self.code_dir.is_dir():
            raise ValueError(f"Code directory is not a directory: {self.code_dir}")
        if not self.script.exists():
            raise ValueError(f"Script does not exist: {self.script}")
        if not self.script.is_file():
            raise ValueError(f"Script is not a file: {self.script}")
        # Check script is within code_dir
        try:
            self.script.relative_to(self.code_dir)
        except ValueError as e:
            raise ValueError(
                f"Script {self.script} must be within code directory {self.code_dir}"
            ) from e
        return self


class StorageConfig(BaseModel):
    """Configuration for GCS storage."""

    bucket: Annotated[str, Field(description="GCS bucket name")]
    job_id: Annotated[str, Field(description="Unique job identifier")]

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, v: str) -> str:
        """Validate bucket name format."""
        # GCS bucket naming rules (simplified)
        pattern = r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$"
        if not re.match(pattern, v):
            raise ValueError(
                f"Invalid bucket name '{v}'. Must be 3-63 chars, lowercase letters, "
                "numbers, hyphens, underscores, and periods."
            )
        return v

    @property
    def checkpoint_dir(self) -> str:
        """Get the GCS path for checkpoints."""
        return f"gs://{self.bucket}/{self.job_id}/ckpt"

    @property
    def log_dir(self) -> str:
        """Get the GCS path for logs."""
        return f"gs://{self.bucket}/{self.job_id}/logs"


class SpotJAXConfig(BaseModel):
    """Complete configuration for a SpotJAX run."""

    tpu: TPUConfig
    run: RunConfig
    storage: StorageConfig

    @classmethod
    def from_cli_args(
        cls,
        script: str,
        tpu_type: str = "v4-8",
        zone: str = "us-central2-b",
        project: str | None = None,
        bucket: str | None = None,
        job_id: str | None = None,
        max_retries: int = 5,
        stream_worker: int = 0,
        code_dir: str | None = None,
    ) -> SpotJAXConfig:
        """Create config from CLI arguments with defaults."""
        # Resolve project
        resolved_project = project or get_default_project()
        if not resolved_project:
            raise ValueError(
                "GCP project not specified. Set --project or GOOGLE_CLOUD_PROJECT env var."
            )

        # Resolve bucket (auto-create name if not specified)
        resolved_bucket = bucket or f"spotjax-{resolved_project}"

        # Resolve job_id
        resolved_job_id = job_id or generate_job_id()

        # Resolve code_dir (default to script's parent directory)
        script_path = Path(script).resolve()
        resolved_code_dir = Path(code_dir).resolve() if code_dir else script_path.parent

        return cls(
            tpu=TPUConfig(
                tpu_type=tpu_type,
                zone=zone,
                project=resolved_project,
            ),
            run=RunConfig(
                script=script_path,
                code_dir=resolved_code_dir,
                max_retries=max_retries,
                stream_worker=stream_worker,
            ),
            storage=StorageConfig(
                bucket=resolved_bucket,
                job_id=resolved_job_id,
            ),
        )

    def get_env_vars(
        self,
        worker_id: int = 0,
        coordinator_ip: str | None = None,
        is_restart: bool = False,
    ) -> dict[str, str]:
        """Get environment variables to inject into remote environment.

        Args:
            worker_id: The worker/node ID (0-indexed)
            coordinator_ip: IP address of the coordinator node (only for multi-node)
            is_restart: Whether this is a restart after preemption

        Returns:
            Dictionary of environment variables
        """
        env = {
            "SPOT_CHECKPOINT_DIR": self.storage.checkpoint_dir,
            "SPOT_LOG_DIR": self.storage.log_dir,
            "SPOT_JOB_ID": self.storage.job_id,
            "SPOT_IS_RESTART": "true" if is_restart else "false",
        }

        # Only add multi-node variables if we have multiple nodes
        if self.tpu.num_nodes > 1:
            env["SPOT_WORKER_ID"] = str(worker_id)
            env["SPOT_NUM_WORKERS"] = str(self.tpu.num_nodes)
            if coordinator_ip:
                env["JAX_COORDINATOR_ADDRESS"] = f"{coordinator_ip}:1234"

        return env
