"""Main orchestration loop for SpotJAX."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import signal
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from spotjax.config import SpotJAXConfig
from spotjax.infra.gcp import TPUProvider, TPUProviderError, generate_tpu_name
from spotjax.infra.ssh import ClusterRunner, SSHConnectionError
from spotjax.infra.storage import ensure_bucket_exists
from spotjax.utils.logging import (
    NodeLoggerManager,
    console,
    get_log_dir,
    print_config,
    print_error,
    print_header,
    print_status,
    print_success,
    print_warning,
)
from spotjax.watchdog import CheckpointWatchdog


class RunResult(Enum):
    """Result of a single run attempt."""

    SUCCESS = "success"  # Script completed with exit code 0
    PREEMPTED = "preempted"  # TPU was preempted or node failed
    SCRIPT_ERROR = "script_error"  # Script exited with non-zero code
    INFRA_ERROR = "infra_error"  # TPU/SSH infrastructure error
    USER_CANCELLED = "user_cancelled"  # User pressed Ctrl+C


@dataclass
class AttemptResult:
    """Result of a single orchestration attempt."""

    result: RunResult
    exit_code: int
    message: str = ""


class Orchestrator:
    """Orchestrates the full lifecycle of a TPU training run.

    Handles:
    1. TPU provisioning via GCP API
    2. SSH connection to all nodes
    3. Code upload via rsync
    4. Script execution with environment variable injection
    5. Output streaming and logging
    6. Automatic retry on preemption
    7. Graceful cleanup on exit or error
    """

    def __init__(self, config: SpotJAXConfig):
        """Initialize orchestrator.

        Args:
            config: Complete SpotJAX configuration
        """
        self.config = config
        self.tpu_name: str | None = None
        self.tpu_provider: TPUProvider | None = None
        self.cluster_runner: ClusterRunner | None = None
        self.node_ips: list[str] = []
        self.log_dir: Path | None = None
        self.logger_manager: NodeLoggerManager | None = None
        self.watchdog: CheckpointWatchdog | None = None

        # Retry state
        self._attempt = 0
        self._max_retries = config.run.max_retries

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()
        self._cleanup_done = asyncio.Event()

    async def run(self) -> int:
        """Run the full orchestration loop with automatic retry.

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        print_header(f"SpotJAX - {self.config.storage.job_id}")

        # Print configuration
        print_status("Configuration:")
        print_config(
            {
                "TPU Type": self.config.tpu.tpu_type,
                "Zone": self.config.tpu.zone,
                "Project": self.config.tpu.project,
                "Script": self.config.run.script,
                "Bucket": self.config.storage.bucket,
                "Job ID": self.config.storage.job_id,
                "Max Retries": self._max_retries,
            }
        )
        console.print()

        # Ensure GCS bucket exists for checkpoints/logs
        try:
            await ensure_bucket_exists(
                bucket_name=self.config.storage.bucket,
                project=self.config.tpu.project,
            )
        except Exception as e:
            print_error(f"Failed to setup GCS bucket: {e}")
            return 1

        # Main retry loop
        while self._attempt <= self._max_retries:
            if self._attempt > 0:
                print_header(f"Retry attempt {self._attempt}/{self._max_retries}") # i am not sure but we run first single attemp so it's shoudnt be a retry what you think?

            result = await self._run_single_attempt() # maybe think on naming strait forward for attemp but don't reflet the function itself

            if result.result == RunResult.SUCCESS:
                print_success("Training completed successfully!")
                return 0

            if result.result == RunResult.USER_CANCELLED:
                print_warning("Run cancelled by user")
                return 130  # Standard Ctrl+C exit code

            # Check if we should retry
            if result.result in (RunResult.PREEMPTED, RunResult.INFRA_ERROR):
                if self._attempt < self._max_retries:
                    self._attempt += 1
                    print_warning(
                        f"{result.message} - will retry ({self._attempt}/{self._max_retries})"
                    )
                    # Small delay before retry
                    await asyncio.sleep(5)
                    continue
                else:
                    print_error(f"Max retries ({self._max_retries}) exceeded")
                    return 1

            if result.result == RunResult.SCRIPT_ERROR:
                # Script error - could be a bug in user code, or could be preemption
                # We retry these too since preemption can cause various exit codes
                if self._attempt < self._max_retries:
                    self._attempt += 1
                    print_warning(
                        f"Script failed (exit code {result.exit_code}) - "
                        f"will retry ({self._attempt}/{self._max_retries})"
                    )
                    await asyncio.sleep(5)
                    continue
                else:
                    print_error(
                        f"Script failed with exit code {result.exit_code} "
                        f"after {self._max_retries} retries"
                    )
                    return result.exit_code

        # Should not reach here
        return 1

    async def _run_single_attempt(self) -> AttemptResult:
        """Run a single attempt of the training job.

        Returns:
            AttemptResult with the outcome
        """
        is_restart = self._attempt > 0

        try:
            # Phase 1: Provision TPU
            await self._provision_tpu()

            # Phase 2: Connect SSH
            await self._connect_ssh()

            # Phase 3: Upload code
            await self._upload_code()

            # Phase 4: Install dependencies
            await self._install_dependencies()

            # Phase 5: Run script
            exit_code = await self._run_script(is_restart=is_restart)

            if self._shutdown_event.is_set():
                return AttemptResult(
                    result=RunResult.USER_CANCELLED,
                    exit_code=130,
                    message="User cancelled",
                )

            if exit_code == 0:
                return AttemptResult(
                    result=RunResult.SUCCESS,
                    exit_code=0,
                    message="Completed successfully",
                )
            else:
                return AttemptResult(
                    result=RunResult.SCRIPT_ERROR,
                    exit_code=exit_code,
                    message=f"Script exited with code {exit_code}",
                )

        except TPUProviderError as e:
            print_error(f"TPU error: {e}")
            return AttemptResult(
                result=RunResult.INFRA_ERROR,
                exit_code=1,
                message=f"TPU error: {e}",
            )
        except SSHConnectionError as e:
            # SSH failures often indicate preemption
            print_error(f"SSH error: {e}")
            return AttemptResult(
                result=RunResult.PREEMPTED,
                exit_code=1,
                message=f"SSH connection lost: {e}",
            )
        except asyncio.CancelledError:
            return AttemptResult(
                result=RunResult.USER_CANCELLED,
                exit_code=130,
                message="Cancelled",
            )
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            return AttemptResult(
                result=RunResult.INFRA_ERROR,
                exit_code=1,
                message=f"Unexpected error: {e}",
            )
        finally:
            await self._cleanup()

    async def _provision_tpu(self) -> None:
        """Provision TPU VM."""
        self.tpu_provider = TPUProvider(
            project=self.config.tpu.project,
            zone=self.config.tpu.zone,
        )

        # Generate a unique TPU name
        self.tpu_name = generate_tpu_name()

        # Create TPU using queued resources
        await self.tpu_provider.create_tpu(
            name=self.tpu_name,
            tpu_type=self.config.tpu.tpu_type,
            spot=self.config.tpu.spot,
        )

        # Wait for TPU to be ready
        ready = await self.tpu_provider.wait_for_ready(
            name=self.tpu_name,
            timeout=3600,  # 1 hour max wait
        )

        if not ready:
            raise TPUProviderError("TPU provisioning timed out")

        # Get node IPs
        # Use external IPs for SSH access
        self.node_ips = await self.tpu_provider.get_node_external_ips(self.tpu_name)
        if len(self.node_ips) == 1:
            print_status("TPU ready")
        else:
            print_status(f"TPU ready with {len(self.node_ips)} nodes")

    async def _connect_ssh(self) -> None:
        """Establish SSH connections to all nodes."""
        self.cluster_runner = ClusterRunner(
            ips=self.node_ips,
            username="root",  # TPU VMs use root
        )

        # Small delay for SSH daemon to be ready
        await asyncio.sleep(5)

        await self.cluster_runner.connect_all()

    async def _upload_code(self) -> None:
        """Upload user code to all nodes."""
        if not self.cluster_runner:
            raise RuntimeError("SSH not connected")

        await self.cluster_runner.upload_directory(
            local_path=self.config.run.code_dir,
            remote_path="/root/code",
        )

    def _compute_requirements_hash(self) -> str | None:
        """Compute SHA256 hash of requirements.txt for cache key.

        Returns:
            Hex string of hash, or None if requirements.txt doesn't exist
        """
        requirements_path = self.config.run.code_dir / "requirements.txt"
        if not requirements_path.exists():
            return None

        content = requirements_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]  # Use first 16 chars

    async def _install_dependencies(self) -> None:
        """Install Python dependencies with GCS caching.

        Uses a hash of requirements.txt as the cache key to ensure different
        projects with different dependencies don't conflict.
        """
        if not self.cluster_runner:
            raise RuntimeError("SSH not connected")

        # Check if requirements.txt exists locally
        requirements_hash = self._compute_requirements_hash()
        if not requirements_hash:
            return  # No requirements.txt, skip installation

        # Cache paths
        cache_dir = "/root/.cache/pip"
        cache_bucket_path = f"gs://{self.config.storage.bucket}/pip-cache/{requirements_hash}"

        # Try to restore pip cache from GCS (helps on retries and across jobs with same deps)
        print_status("Checking for cached pip packages...")
        cache_restore_results = await self.cluster_runner.run_command(
            f"gsutil -m rsync -r {cache_bucket_path}/ {cache_dir}/ 2>/dev/null || true",
            timeout=300,  # 5 minute timeout for cache restore
        )

        # Check if cache was found
        cache_hit = any(
            "Copying" in result.stdout or "Copying" in result.stderr
            for result in cache_restore_results.values()
        )
        if cache_hit:
            print_status("Restored pip cache from GCS", style="success")

        # Run pip install on all nodes (will use cache if available)
        print_status("Installing dependencies from requirements.txt...")
        results = await self.cluster_runner.run_command(
            "pip install -q -r /root/code/requirements.txt",
            timeout=1200,  # 20 minute timeout for pip install
        )

        # Check results
        failed_nodes = [
            node_id for node_id, result in results.items() if not result.success
        ]

        if failed_nodes:
            # Log errors but continue - some packages might be pre-installed
            for node_id in failed_nodes:
                if len(results) == 1:
                    print_warning(f"pip install warning: {results[node_id].stderr[:200]}")
                else:
                    print_warning(
                        f"Node {node_id} pip install warning: {results[node_id].stderr[:200]}"
                    )
        else:
            print_status("Dependencies installed", style="success")

            # Save cache to GCS for future runs (only from node 0 to avoid redundant uploads)
            if not cache_hit:  # Only upload if we didn't restore from cache
                print_status("Caching pip packages to GCS for future runs...")
                cache_save_result = await self.cluster_runner.run_command(
                    f"gsutil -m rsync -r {cache_dir}/ {cache_bucket_path}/ 2>/dev/null || true",
                    node=0,  # Only need to cache from one node
                    timeout=300,  # 5 minute timeout for cache save
                )
                if cache_save_result[0].exit_code == 0:
                    print_status("Pip cache saved to GCS", style="success")

    async def _run_script(self, is_restart: bool = False) -> int:
        """Run the training script on all nodes.

        Args:
            is_restart: Whether this is a restart after preemption

        Returns:
            Exit code from node 0 (coordinator)
        """
        if not self.cluster_runner:
            raise RuntimeError("SSH not connected")

        # Setup logging
        self.log_dir = get_log_dir()
        self.logger_manager = NodeLoggerManager(
            num_nodes=len(self.node_ips),
            log_dir=self.log_dir,
            stream_worker=self.config.run.stream_worker,
        )

        attempt_info = f" (attempt {self._attempt + 1})" if self._attempt > 0 else ""
        is_single_node = len(self.node_ips) == 1

        print_status(f"Logs directory: {self.log_dir}")
        if not is_single_node:
            print_status(f"Streaming output from worker {self.config.run.stream_worker}")
        if is_restart:
            print_status("SPOT_IS_RESTART=true (resuming from checkpoint)")

        # Start checkpoint watchdog
        self.watchdog = CheckpointWatchdog(
            checkpoint_dir=self.config.storage.checkpoint_dir,
            stall_threshold_minutes=60,
            poll_interval_minutes=5,
        )
        self.watchdog.start()
        print_status(
            f"Checkpoint watchdog started{attempt_info} (warns if no checkpoint for 60 min)"
        )
        console.print()

        # Build the command
        script_relative = self.config.run.script.relative_to(self.config.run.code_dir)

        # Build environment variables for each node
        coordinator_ip = self.node_ips[0]

        with self.logger_manager:
            # Callbacks for streaming output
            def stdout_callback(node_id: int, line: str) -> None:
                if self.logger_manager:
                    self.logger_manager.get(node_id).write(line)

            def stderr_callback(node_id: int, line: str) -> None:
                if self.logger_manager:
                    self.logger_manager.get(node_id).write_stderr(line)

            async def run_on_node(node_id: int) -> int:
                env = self.config.get_env_vars(
                    worker_id=node_id,
                    coordinator_ip=coordinator_ip,
                    is_restart=is_restart,
                )

                # Build command with env vars
                env_str = " ".join(f'{k}="{v}"' for k, v in env.items())
                full_cmd = f"cd /root/code && {env_str} python {script_relative}"

                conn = self.cluster_runner._connections[node_id]

                try:
                    async with conn.create_process(full_cmd) as process:

                        async def read_stdout():
                            try:
                                async for line in process.stdout:
                                    stdout_callback(node_id, line.rstrip("\n"))
                            except Exception:
                                pass

                        async def read_stderr():
                            try:
                                async for line in process.stderr:
                                    stderr_callback(node_id, line.rstrip("\n"))
                            except Exception:
                                pass

                        # Monitor for shutdown while running
                        async def wait_for_shutdown():
                            await self._shutdown_event.wait()
                            # Kill remote process
                            with contextlib.suppress(Exception):
                                process.terminate()

                        # Run all tasks concurrently
                        done, pending = await asyncio.wait(
                            [
                                asyncio.create_task(read_stdout()),
                                asyncio.create_task(read_stderr()),
                                asyncio.create_task(process.wait()),
                                asyncio.create_task(wait_for_shutdown()),
                            ],
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        # Cancel pending tasks
                        for task in pending:
                            task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task

                        # If shutdown was triggered, return special code
                        if self._shutdown_event.is_set():
                            return 130

                        return process.exit_status or 0

                except Exception as e:
                    stderr_callback(node_id, f"Error: {e}")
                    # Connection errors during execution indicate preemption
                    raise SSHConnectionError(f"Node {node_id} connection lost: {e}") from e

            # Run on all nodes concurrently
            tasks = [
                asyncio.create_task(run_on_node(i)) for i in range(len(self.node_ips))
            ]

            try:
                exit_codes = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                # If any node fails, cancel all others
                for task in tasks:
                    task.cancel()
                raise

            # Check for exceptions (indicates node failure/preemption)
            for i, result in enumerate(exit_codes):
                if isinstance(result, Exception):
                    raise SSHConnectionError(
                        f"Node {i} failed: {result}"
                    ) from result

            # Log file locations
            console.print()
            print_status(f"Logs written to: {self.log_dir}")

            # Return exit code from node 0 (coordinator)
            return exit_codes[0]

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        def signal_handler(sig: signal.Signals) -> None:
            print_warning(f"\nReceived {sig.name}, shutting down...")
            self._shutdown_event.set()

        # Handle SIGINT (Ctrl+C) and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler, sig)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, _f: signal_handler(signal.Signals(s)))

    async def _cleanup(self) -> None:
        """Clean up resources for this attempt."""
        print_status("Cleaning up...")

        # Stop watchdog
        if self.watchdog:
            await self.watchdog.stop()
            self.watchdog = None

        # Close SSH connections
        if self.cluster_runner:
            try:
                await self.cluster_runner.close()
            except Exception as e:
                print_warning(f"Error closing SSH: {e}")
            self.cluster_runner = None

        # Delete TPU
        if self.tpu_provider and self.tpu_name:
            try:
                await self.tpu_provider.delete_tpu(self.tpu_name)
            except Exception as e:
                print_warning(f"Error deleting TPU: {e}")
            self.tpu_name = None

        # Reset node IPs for next attempt
        self.node_ips = []

        self._cleanup_done.set()
        print_success("Cleanup complete")


async def run_orchestrator(config: SpotJAXConfig) -> int:
    """Run the orchestrator with the given config.

    Args:
        config: SpotJAX configuration

    Returns:
        Exit code
    """
    orchestrator = Orchestrator(config)
    return await orchestrator.run()
