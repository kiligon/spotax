"""SpotJAX utilities for checkpoint management and preemption handling.

Copy this file to your project and import it in your training script.
This file has no runtime dependency on the spotax package.

Usage:
    from spotax_utils import CheckpointManager, get_config

    config = get_config()
    ckpt = CheckpointManager(config.checkpoint_dir)

    state, start_step = ckpt.restore_or_init(initial_state)

    for step in range(start_step, max_steps):
        state = train_step(state, batch)
        ckpt.save(step, state)  # Async save, respects save_interval_steps

        # Exit gracefully on preemption (checkpoint already saved by Orbax)
        if ckpt.reached_preemption(step):
            break

Orbax Autocheckpoint:
    SpotJAX enables GCP's autocheckpoint feature. When preemption occurs:
    1. GCP sends SIGTERM to the TPU VM
    2. Orbax automatically saves a checkpoint (even if not at save_interval)
    3. reached_preemption() returns True so you can exit cleanly
    4. Orchestrator detects the exit and retries with a new TPU

Environment variables (set by SpotJAX CLI):
    SPOT_CHECKPOINT_DIR - GCS path for checkpoints (required)
    SPOT_LOG_DIR - GCS path for logs
    SPOT_JOB_ID - Unique job identifier
    SPOT_IS_RESTART - "true" if resuming after preemption

For multi-node TPU (v4-16+), also set automatically:
    SPOT_WORKER_ID, SPOT_NUM_WORKERS, JAX_COORDINATOR_ADDRESS
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
from dataclasses import dataclass
from typing import Any, TypeVar

import jax
import orbax.checkpoint as ocp

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class SpotConfig:
    """Configuration from SpotJAX environment variables."""

    checkpoint_dir: str
    log_dir: str
    job_id: str
    is_restart: bool
    worker_id: int = 0
    num_workers: int = 1
    coordinator_address: str | None = None

    @property
    def is_multi_node(self) -> bool:
        """True if running on multiple TPU VMs (v4-16+)."""
        return self.num_workers > 1

    @property
    def is_coordinator(self) -> bool:
        """True if this is the coordinator node (always True for single-node)."""
        return self.worker_id == 0


def get_config() -> SpotConfig:
    """Load configuration from environment variables."""
    checkpoint_dir = os.environ.get("SPOT_CHECKPOINT_DIR", "")
    if not checkpoint_dir:
        raise ValueError("SPOT_CHECKPOINT_DIR not set. Run with: spotax run ...")

    return SpotConfig(
        checkpoint_dir=checkpoint_dir,
        log_dir=os.environ.get("SPOT_LOG_DIR", ""),
        job_id=os.environ.get("SPOT_JOB_ID", "unknown"),
        is_restart=os.environ.get("SPOT_IS_RESTART", "").lower() == "true",
        worker_id=int(os.environ.get("SPOT_WORKER_ID", "0")),
        num_workers=int(os.environ.get("SPOT_NUM_WORKERS", "1")),
        coordinator_address=os.environ.get("JAX_COORDINATOR_ADDRESS"),
    )


def setup_distributed(config: SpotConfig | None = None) -> None:
    """Initialize JAX distributed runtime for multi-node training.

    Only needed for v4-16+ (multiple TPU VMs). For v4-8 (single VM),
    this is a no-op - JAX automatically uses all local TPU chips.
    """
    if config is None:
        config = get_config()

    if not config.is_multi_node:
        # Single node - JAX uses all local TPU chips automatically
        logger.info(f"Single-node mode: {jax.device_count()} devices")
        return

    if not config.coordinator_address:
        raise ValueError("JAX_COORDINATOR_ADDRESS not set for multi-node training")

    jax.distributed.initialize(
        coordinator_address=config.coordinator_address,
        num_processes=config.num_workers,
        process_id=config.worker_id,
    )
    logger.info(f"Multi-node mode: {jax.device_count()} devices across {jax.process_count()} hosts")


class CheckpointManager:
    """Checkpoint manager with Orbax autocheckpoint support for Spot preemption.

    Uses Orbax's built-in preemption detection which automatically saves
    a checkpoint when SIGTERM is received (GCP autocheckpoint feature).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 3,
        save_interval_steps: int = 1000,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: GCS path for checkpoints (e.g., gs://bucket/checkpoints)
            max_to_keep: Maximum number of checkpoints to retain
            save_interval_steps: Save checkpoint every N steps. Orbax will also
                save automatically on preemption regardless of this interval.
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_interval_steps = save_interval_steps
        self._manager = ocp.CheckpointManager(
            directory=checkpoint_dir,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                save_interval_steps=save_interval_steps,
            ),
            item_names=("state",),
        )
        self._current_state: Any = None
        self._current_step: int = 0
        logger.info(f"CheckpointManager: {checkpoint_dir} (save every {save_interval_steps} steps)")

    @property
    def latest_step(self) -> int | None:
        """Latest checkpoint step, or None if no checkpoints."""
        return self._manager.latest_step()

    def save(self, step: int, state: T) -> bool:
        """Save checkpoint if at save interval (or on preemption).

        Orbax handles save_interval_steps internally and will skip saves
        that aren't at the interval. On preemption, Orbax saves regardless.

        Args:
            step: Current training step
            state: Model state to checkpoint

        Returns:
            True if a checkpoint was saved, False if skipped (not at interval)
        """
        self._current_state = state
        self._current_step = step
        saved = self._manager.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(state)))
        if saved:
            logger.info(f"Saved checkpoint at step {step}")
        return saved

    def restore(self, step: int | None = None) -> tuple[Any, int]:
        """Restore checkpoint. Returns (state, step)."""
        if step is None:
            step = self.latest_step
        if step is None:
            raise ValueError(f"No checkpoints in {self.checkpoint_dir}")

        result = self._manager.restore(step, args=ocp.args.Composite(state=ocp.args.StandardRestore()))
        logger.info(f"Restored checkpoint from step {step}")
        return result["state"], step

    def restore_or_init(self, init_state: T, init_step: int = 0) -> tuple[T, int]:
        """Restore latest checkpoint, or return init_state if none exists."""
        if self.latest_step is not None:
            return self.restore()
        return init_state, init_step

    def reached_preemption(self, step: int) -> bool:
        """Check if preemption signal was received.

        Orbax uses JAX multihost_utils to detect SIGTERM across all hosts.
        When preemption is detected, Orbax automatically saves a checkpoint
        (even if not at save_interval_steps).

        Call this after save() in your training loop to exit gracefully:

            if ckpt.reached_preemption(step):
                break  # Exit loop, orchestrator will retry

        Args:
            step: Current training step

        Returns:
            True if preemption was detected and checkpoint saved
        """
        if self._manager.reached_preemption(step):
            logger.warning(f"Preemption detected at step {step} - exiting after checkpoint save")
            self._manager.wait_until_finished()
            return True
        return False

    def setup_preemption_handler(self) -> None:
        """Register SIGTERM handler for graceful Spot preemption.

        NOTE: This is a legacy fallback. The recommended approach is to use
        reached_preemption() in your training loop, which uses Orbax's built-in
        preemption detection with proper multi-host coordination.

        When preempted, saves current state before exit.
        Call this after your first checkpoint save.
        """
        def on_sigterm(_sig: int, _frame: Any) -> None:
            logger.warning("SIGTERM received - saving checkpoint before exit")
            if self._current_state is not None:
                try:
                    self._manager.save(
                        self._current_step,
                        args=ocp.args.Composite(state=ocp.args.StandardSave(self._current_state)),
                    )
                    self._manager.wait_until_finished()
                    logger.info(f"Emergency save completed at step {self._current_step}")
                except Exception as e:
                    logger.error(f"Emergency save failed: {e}")
            sys.exit(0)

        signal.signal(signal.SIGTERM, on_sigterm)
        atexit.register(self._manager.wait_until_finished)
        logger.info("Preemption handler registered (legacy mode)")

    def wait(self) -> None:
        """Wait for pending saves to complete."""
        self._manager.wait_until_finished()

    def close(self) -> None:
        """Close the checkpoint manager."""
        self.wait()
        self._manager.close()
