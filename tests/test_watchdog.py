"""Tests for checkpoint watchdog."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from spotjax.watchdog import CheckpointWatchdog


class TestCheckpointWatchdogInit:
    """Tests for CheckpointWatchdog initialization."""

    def test_parses_gcs_path(self):
        """Test that GCS path is parsed correctly."""
        watchdog = CheckpointWatchdog("gs://my-bucket/path/to/ckpt")
        assert watchdog.bucket_name == "my-bucket"
        assert watchdog.prefix == "path/to/ckpt"

    def test_parses_bucket_only_path(self):
        """Test parsing GCS path with no prefix."""
        watchdog = CheckpointWatchdog("gs://my-bucket")
        assert watchdog.bucket_name == "my-bucket"
        assert watchdog.prefix == ""

    def test_raises_for_non_gcs_path(self):
        """Test that non-GCS paths raise an error."""
        with pytest.raises(ValueError, match="must be a GCS path"):
            CheckpointWatchdog("/local/path")

    def test_default_thresholds(self):
        """Test default threshold values."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")
        assert watchdog.stall_threshold_seconds == 60 * 60  # 60 minutes
        assert watchdog.poll_interval_seconds == 5 * 60  # 5 minutes

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        watchdog = CheckpointWatchdog(
            "gs://bucket/prefix",
            stall_threshold_minutes=30,
            poll_interval_minutes=10,
        )
        assert watchdog.stall_threshold_seconds == 30 * 60
        assert watchdog.poll_interval_seconds == 10 * 60


class TestCheckpointWatchdogProperties:
    """Tests for CheckpointWatchdog properties."""

    def test_is_running_false_initially(self):
        """Test that watchdog is not running initially."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")
        assert watchdog.is_running is False

    def test_last_checkpoint_info_empty_initially(self):
        """Test last checkpoint info when no checkpoint seen."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")
        info = watchdog.last_checkpoint_info
        assert info["name"] is None
        assert info["time"] is None
        assert info["minutes_ago"] is None


class TestCheckpointWatchdogLifecycle:
    """Tests for watchdog start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        """Test that start creates background task."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")
        watchdog.start()
        try:
            assert watchdog._task is not None
            assert watchdog.is_running is True
        finally:
            await watchdog.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """Test that stop cancels the background task."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")
        watchdog.start()
        await watchdog.stop()
        assert watchdog._task is None
        assert watchdog.is_running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        """Test that calling start twice doesn't create duplicate tasks."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")
        watchdog.start()
        task1 = watchdog._task
        watchdog.start()  # Should be a no-op
        task2 = watchdog._task
        assert task1 is task2
        await watchdog.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self):
        """Test that calling stop twice is safe."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")
        await watchdog.stop()  # Should be a no-op
        await watchdog.stop()  # Should still be a no-op


class TestCheckpointWatchdogCheckOnce:
    """Tests for the _check_once method."""

    @pytest.mark.asyncio
    async def test_handles_no_checkpoints(self):
        """Test behavior when no checkpoints exist."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")

        with patch.object(watchdog, "_get_latest_checkpoint") as mock_get:
            mock_get.return_value = (None, None)
            await watchdog._check_once()

        # Should not raise, should not set warning
        assert watchdog._last_checkpoint_time is None
        assert watchdog._warning_shown is False

    @pytest.mark.asyncio
    async def test_updates_on_new_checkpoint(self):
        """Test that state updates when a new checkpoint is found."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")

        now = datetime.now(timezone.utc)
        with patch.object(watchdog, "_get_latest_checkpoint") as mock_get:
            mock_get.return_value = ("prefix/ckpt-100", now)
            await watchdog._check_once()

        assert watchdog._last_checkpoint_time == now
        assert watchdog._last_checkpoint_name == "prefix/ckpt-100"

    @pytest.mark.asyncio
    async def test_handles_gcs_errors_gracefully(self):
        """Test that GCS errors don't crash the watchdog."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")

        with patch.object(watchdog, "_get_latest_checkpoint") as mock_get:
            mock_get.side_effect = Exception("GCS error")
            # Should not raise
            await watchdog._check_once()


class TestCheckpointWatchdogStallDetection:
    """Tests for stall detection logic."""

    @pytest.mark.asyncio
    async def test_warns_when_stalled(self):
        """Test that warning is shown when no new checkpoint for threshold time."""
        watchdog = CheckpointWatchdog(
            "gs://bucket/prefix",
            stall_threshold_minutes=1,  # 1 minute
        )

        # Set last checkpoint to 2 minutes ago
        from datetime import timedelta

        old_time = datetime.now(timezone.utc) - timedelta(minutes=2)
        watchdog._last_checkpoint_time = old_time
        watchdog._last_checkpoint_name = "prefix/ckpt-old"

        with patch.object(watchdog, "_get_latest_checkpoint") as mock_get:
            # Return the same old checkpoint
            mock_get.return_value = ("prefix/ckpt-old", old_time)

            with patch("spotjax.watchdog.print_warning") as mock_warn:
                await watchdog._check_once()
                mock_warn.assert_called_once()
                assert "No new checkpoint" in mock_warn.call_args[0][0]

        assert watchdog._warning_shown is True

    @pytest.mark.asyncio
    async def test_warning_shown_once(self):
        """Test that warning is only shown once per stall."""
        watchdog = CheckpointWatchdog(
            "gs://bucket/prefix",
            stall_threshold_minutes=1,
        )

        # Set last checkpoint to 2 minutes ago
        from datetime import timedelta

        old_time = datetime.now(timezone.utc) - timedelta(minutes=2)
        watchdog._last_checkpoint_time = old_time
        watchdog._last_checkpoint_name = "prefix/ckpt-old"

        with patch.object(watchdog, "_get_latest_checkpoint") as mock_get:
            mock_get.return_value = ("prefix/ckpt-old", old_time)

            with patch("spotjax.watchdog.print_warning") as mock_warn:
                # First check shows warning
                await watchdog._check_once()
                assert mock_warn.call_count == 1

                # Second check should not show warning again
                await watchdog._check_once()
                assert mock_warn.call_count == 1

    @pytest.mark.asyncio
    async def test_warning_resets_on_new_checkpoint(self):
        """Test that warning flag resets when new checkpoint appears."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")
        watchdog._warning_shown = True

        now = datetime.now(timezone.utc)
        with patch.object(watchdog, "_get_latest_checkpoint") as mock_get:
            mock_get.return_value = ("prefix/ckpt-new", now)
            await watchdog._check_once()

        assert watchdog._warning_shown is False


class TestCheckpointWatchdogLastCheckpointInfo:
    """Tests for last_checkpoint_info property."""

    def test_returns_none_values_when_no_checkpoint(self):
        """Test info when no checkpoint has been seen."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")
        info = watchdog.last_checkpoint_info

        assert info["name"] is None
        assert info["time"] is None
        assert info["minutes_ago"] is None

    def test_returns_info_when_checkpoint_exists(self):
        """Test info after checkpoint has been recorded."""
        watchdog = CheckpointWatchdog("gs://bucket/prefix")

        from datetime import timedelta

        # Set checkpoint from 5 minutes ago
        checkpoint_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        watchdog._last_checkpoint_time = checkpoint_time
        watchdog._last_checkpoint_name = "prefix/ckpt-100"

        info = watchdog.last_checkpoint_info

        assert info["name"] == "prefix/ckpt-100"
        assert info["time"] == checkpoint_time.isoformat()
        # Allow 1 minute tolerance for test execution time
        assert 4 <= info["minutes_ago"] <= 6
