"""Checkpoint watchdog for detecting stalled training jobs."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone

from google.cloud import storage

from spotjax.utils.logging import console, print_warning


class CheckpointWatchdog:
    """Monitors checkpoint directory for signs of stalled training.

    Periodically checks GCS for new checkpoint files. If no new checkpoints
    appear within the threshold time, warns the user that the job may be stuck.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        stall_threshold_minutes: int = 60,
        poll_interval_minutes: int = 5,
    ):
        """Initialize the watchdog.

        Args:
            checkpoint_dir: GCS path (gs://bucket/path/to/ckpt)
            stall_threshold_minutes: Warn if no checkpoint for this many minutes
            poll_interval_minutes: How often to check for new checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.stall_threshold_seconds = stall_threshold_minutes * 60
        self.poll_interval_seconds = poll_interval_minutes * 60

        # Parse GCS path
        if not checkpoint_dir.startswith("gs://"):
            raise ValueError(f"checkpoint_dir must be a GCS path: {checkpoint_dir}")

        path = checkpoint_dir[5:]  # Remove "gs://"
        parts = path.split("/", 1)
        self.bucket_name = parts[0]
        self.prefix = parts[1] if len(parts) > 1 else ""

        # State
        self._last_checkpoint_time: datetime | None = None
        self._last_checkpoint_name: str | None = None
        self._warning_shown = False
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        # GCS client (lazy init)
        self._storage_client: storage.Client | None = None

    @property
    def storage_client(self) -> storage.Client:
        """Get or create GCS client."""
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    async def _get_latest_checkpoint(self) -> tuple[str | None, datetime | None]:
        """Get the latest checkpoint from GCS.

        Returns:
            Tuple of (checkpoint_name, modified_time) or (None, None) if no checkpoints.
        """
        def _list_blobs():
            bucket = self.storage_client.bucket(self.bucket_name)
            # List blobs with prefix, get most recently modified
            blobs = list(bucket.list_blobs(prefix=self.prefix, max_results=100))
            if not blobs:
                return None, None

            # Find the most recently modified blob
            latest = max(blobs, key=lambda b: b.updated or datetime.min.replace(tzinfo=timezone.utc))
            return latest.name, latest.updated

        # Run blocking GCS call in thread
        return await asyncio.to_thread(_list_blobs)

    async def _check_once(self) -> None:
        """Perform a single checkpoint check."""
        try:
            name, modified_time = await self._get_latest_checkpoint()

            if name is None:
                # No checkpoints yet - that's normal at the start
                if self._last_checkpoint_time is None:
                    return
            else:
                # Update if we found a newer checkpoint
                if self._last_checkpoint_time is None or modified_time > self._last_checkpoint_time:
                    self._last_checkpoint_time = modified_time
                    self._last_checkpoint_name = name
                    self._warning_shown = False  # Reset warning on new checkpoint
                    console.print(
                        f"  [dim]Watchdog: checkpoint detected ({name.split('/')[-1]})[/dim]"
                    )
                    return

            # Check if we've exceeded the stall threshold
            if self._last_checkpoint_time is not None:
                now = datetime.now(timezone.utc)
                elapsed = (now - self._last_checkpoint_time).total_seconds()

                if elapsed > self.stall_threshold_seconds and not self._warning_shown:
                    minutes_ago = int(elapsed / 60)
                    print_warning(
                        f"No new checkpoint in {minutes_ago} minutes. "
                        f"Job may be stuck. Last: {self._last_checkpoint_name}"
                    )
                    self._warning_shown = True

        except Exception as e:
            # Don't crash on watchdog errors, just log
            console.print(f"  [dim]Watchdog error: {e}[/dim]")

    async def _run_loop(self) -> None:
        """Main watchdog loop."""
        # Initial delay before first check (let training start)
        await asyncio.sleep(60)

        while not self._stop_event.is_set():
            await self._check_once()

            # Wait for next poll, but allow early exit
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.poll_interval_seconds,
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue loop

    def start(self) -> None:
        """Start the watchdog background task."""
        if self._task is not None:
            return  # Already running

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the watchdog."""
        if self._task is None:
            return

        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5)
        except asyncio.TimeoutError:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    @property
    def is_running(self) -> bool:
        """Check if watchdog is running."""
        return self._task is not None and not self._task.done()

    @property
    def last_checkpoint_info(self) -> dict:
        """Get info about the last seen checkpoint."""
        if self._last_checkpoint_time is None:
            return {"name": None, "time": None, "minutes_ago": None}

        now = datetime.now(timezone.utc)
        elapsed = (now - self._last_checkpoint_time).total_seconds()

        return {
            "name": self._last_checkpoint_name,
            "time": self._last_checkpoint_time.isoformat(),
            "minutes_ago": int(elapsed / 60),
        }
