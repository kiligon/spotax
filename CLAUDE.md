# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies (uses uv)
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_config.py

# Run a specific test
uv run pytest tests/test_config.py::TestSpotJAXConfig::test_get_env_vars -v

# Lint code
uv run ruff check src/ tests/

# Lint and auto-fix
uv run ruff check src/ tests/ --fix

# Run the CLI
uv run spotax --help
uv run spotax setup          # Check prerequisites
uv run spotax setup --fix    # Auto-fix prerequisites
uv run spotax run --help
```

## Architecture

SpotJAX orchestrates JAX training jobs on Google Cloud Spot TPUs. The core flow is:

```
CLI (main.py) → Orchestrator → TPUProvider (provision TPU)
                            → ClusterRunner (SSH to nodes)
                            → Upload code via rsync
                            → Install deps (pip install -r requirements.txt)
                            → Run script on all nodes
                            → Watchdog monitors checkpoints
                            → Cleanup on exit/preemption
                            → Retry loop on failure
```

### Key Modules

- **`orchestrator.py`**: Main async loop with retry logic. Handles the full lifecycle: bucket setup → provision → SSH → upload → install deps → run → cleanup. On preemption/failure, cleans up and retries up to `max_retries` times.

- **`infra/gcp.py`**: `TPUProvider` class wraps Google Cloud TPU API (Queued Resources). All GCP SDK calls use `asyncio.to_thread()` since the SDK is synchronous.

- **`infra/ssh.py`**: `ClusterRunner` manages async SSH connections to all TPU nodes using `asyncssh`. Handles parallel command execution and rsync uploads.

- **`infra/storage.py`**: `StorageProvider` for GCS bucket operations. Auto-creates bucket if it doesn't exist.

- **`watchdog.py`**: `CheckpointWatchdog` monitors GCS for new checkpoints. Warns if training appears stalled (no checkpoint for 60 min).

- **`setup.py`**: Prerequisites checker. Validates gcloud, ADC, SSH keys, OS Login. The `spotax setup` command uses this.

- **`config.py`**: Pydantic models (`TPUConfig`, `RunConfig`, `StorageConfig`, `SpotJAXConfig`). Validates TPU types, zones, bucket names.

### Environment Variable Contract

The CLI injects these env vars into remote TPU nodes:
- `SPOT_CHECKPOINT_DIR`: GCS path for checkpoints
- `SPOT_IS_RESTART`: "true" on retry after preemption
- `SPOT_WORKER_ID`: Node index (0 to N-1)
- `SPOT_NUM_WORKERS`: Total node count
- `JAX_COORDINATOR_ADDRESS`: IP:port for JAX distributed

User code reads these via `spotax_utils.py` (copied from examples/).

### Async Patterns

- GCP API calls: wrapped in `asyncio.to_thread()` to avoid blocking SSH keepalives
- SSH: native async via `asyncssh`
- rsync: `asyncio.create_subprocess_exec()`
- Graceful shutdown: `asyncio.Event` for SIGINT/SIGTERM handling
