"""Tests for SpotJAX configuration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from spotax.config import (
    RunConfig,
    SpotJAXConfig,
    StorageConfig,
    TPUConfig,
    generate_job_id,
    get_default_project,
)


class TestTPUConfig:
    """Tests for TPUConfig."""

    def test_valid_tpu_types(self):
        """Test valid TPU type formats."""
        valid_types = ["v4-8", "v4-16", "v4-32", "v4-64", "v4-128", "v5p-8", "v5p-16", "v5litepod-1", "v5litepod-8"]
        for tpu_type in valid_types:
            config = TPUConfig(tpu_type=tpu_type, zone="us-central1-a", project="test-project")
            assert config.tpu_type == tpu_type

    def test_invalid_tpu_type(self):
        """Test invalid TPU type raises error."""
        with pytest.raises(ValueError, match="Invalid TPU type"):
            TPUConfig(tpu_type="invalid-type", zone="us-central1-a", project="test-project")

    def test_valid_zone_format(self):
        """Test valid zone format."""
        config = TPUConfig(tpu_type="v4-8", zone="us-central1-a", project="test-project")
        assert config.zone == "us-central1-a"

    def test_invalid_zone_format(self):
        """Test invalid zone format raises error."""
        with pytest.raises(ValueError, match="Invalid zone format"):
            TPUConfig(tpu_type="v4-8", zone="invalid-zone", project="test-project")

    def test_num_nodes_calculation(self):
        """Test number of nodes calculation."""
        test_cases = [
            ("v4-8", 2),  # 8 chips / 4 = 2 nodes
            ("v4-16", 4),
            ("v4-32", 8),
            ("v4-64", 16),
            ("v5p-8", 2),
        ]
        for tpu_type, expected_nodes in test_cases:
            config = TPUConfig(tpu_type=tpu_type, zone="us-central1-a", project="test-project")
            assert config.num_nodes == expected_nodes, f"Failed for {tpu_type}"


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_valid_bucket_name(self):
        """Test valid bucket names."""
        config = StorageConfig(bucket="my-bucket-123", job_id="20260109-143022")
        assert config.bucket == "my-bucket-123"

    def test_invalid_bucket_name(self):
        """Test invalid bucket name raises error."""
        with pytest.raises(ValueError, match="Invalid bucket name"):
            StorageConfig(bucket="Invalid Bucket!", job_id="20260109-143022")

    def test_checkpoint_dir(self):
        """Test checkpoint directory path generation."""
        config = StorageConfig(bucket="my-bucket", job_id="20260109-143022")
        assert config.checkpoint_dir == "gs://my-bucket/20260109-143022/ckpt"

    def test_log_dir(self):
        """Test log directory path generation."""
        config = StorageConfig(bucket="my-bucket", job_id="20260109-143022")
        assert config.log_dir == "gs://my-bucket/20260109-143022/logs"


class TestRunConfig:
    """Tests for RunConfig."""

    def test_valid_config(self):
        """Test valid run configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "train.py"
            script_path.write_text("print('hello')")

            config = RunConfig(
                script=script_path,
                code_dir=tmpdir,
                max_retries=5,
                stream_worker=0,
            )
            assert config.script == script_path
            assert config.code_dir == Path(tmpdir)

    def test_script_must_exist(self):
        """Test that script must exist."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(
            ValueError, match="Script does not exist"
        ):
            RunConfig(
                script="/nonexistent/script.py",
                code_dir=tmpdir,
            )

    def test_script_must_be_in_code_dir(self):
        """Test that script must be within code directory."""
        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            script_path = Path(tmpdir1) / "train.py"
            script_path.write_text("print('hello')")

            with pytest.raises(ValueError, match="must be within code directory"):
                RunConfig(
                    script=script_path,
                    code_dir=tmpdir2,
                )


class TestSpotJAXConfig:
    """Tests for SpotJAXConfig."""

    def test_from_cli_args(self):
        """Test creating config from CLI arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "train.py"
            script_path.write_text("print('hello')")

            config = SpotJAXConfig.from_cli_args(
                script=str(script_path),
                tpu_type="v4-8",
                zone="us-central1-a",
                project="test-project",
                bucket="test-bucket",
                job_id="test-job",
            )

            assert config.tpu.tpu_type == "v4-8"
            assert config.tpu.zone == "us-central1-a"
            assert config.tpu.project == "test-project"
            assert config.storage.bucket == "test-bucket"
            assert config.storage.job_id == "test-job"
            assert config.run.script == script_path

    def test_get_env_vars(self):
        """Test environment variable generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "train.py"
            script_path.write_text("print('hello')")

            config = SpotJAXConfig.from_cli_args(
                script=str(script_path),
                project="test-project",
                bucket="test-bucket",
                job_id="test-job",
            )

            env = config.get_env_vars()

            assert env["SPOT_CHECKPOINT_DIR"] == "gs://test-bucket/test-job/ckpt"
            assert env["SPOT_LOG_DIR"] == "gs://test-bucket/test-job/logs"
            assert env["SPOT_JOB_ID"] == "test-job"
            assert env["SPOT_IS_RESTART"] == "false"

            # JAX auto-discovers TPU topology, no distributed vars needed
            assert "SPOT_WORKER_ID" not in env
            assert "SPOT_NUM_WORKERS" not in env
            assert "JAX_COORDINATOR_ADDRESS" not in env

    def test_get_env_vars_is_restart(self):
        """Test environment variable generation with is_restart=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "train.py"
            script_path.write_text("print('hello')")

            config = SpotJAXConfig.from_cli_args(
                script=str(script_path),
                project="test-project",
                bucket="test-bucket",
                job_id="test-job",
            )

            env_first_run = config.get_env_vars(is_restart=False)
            assert env_first_run["SPOT_IS_RESTART"] == "false"

            env_restart = config.get_env_vars(is_restart=True)
            assert env_restart["SPOT_IS_RESTART"] == "true"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_job_id_format(self):
        """Test job ID format."""
        job_id = generate_job_id()
        # Should be in YYYYMMDD-HHMMSS format
        assert len(job_id) == 15
        assert job_id[8] == "-"

    def test_get_default_project_from_env(self):
        """Test getting project from environment."""
        # Save original
        original = os.environ.get("GOOGLE_CLOUD_PROJECT")

        try:
            os.environ["GOOGLE_CLOUD_PROJECT"] = "env-project"
            assert get_default_project() == "env-project"
        finally:
            # Restore
            if original:
                os.environ["GOOGLE_CLOUD_PROJECT"] = original
            else:
                os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
