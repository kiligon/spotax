# SpotJAX

CLI tool for running JAX training on Google Cloud Spot TPUs with automatic preemption recovery. Provisions TPUs, uploads code, runs training, and seamlessly retries when Spot instances get preempted.

## Installation

```bash
pip install spotax
```

Requires Python 3.10+ and [gcloud CLI](https://cloud.google.com/sdk/docs/install).

```bash
# Verify prerequisites
spotax setup

# Auto-fix issues (SSH keys, OS Login)
spotax setup --fix
```

## Quick Start

```bash
spotax run train.py --tpu v5litepod-1 --zone us-central1-a
```

SpotJAX will:
1. Create a GCS bucket for checkpoints
2. Provision a Spot TPU
3. SSH into all nodes and upload your code via rsync
4. Run `spotax_setup.sh` if present (custom pre-install steps)
5. Install `requirements.txt` dependencies
6. Run your script with checkpoint/distributed env vars injected
7. On preemption: clean up, provision a new TPU, and resume from last checkpoint

## Project Structure

```
your-project/
  train.py              # Your training script
  data.py               # Data loading (optional)
  spotax_utils.py       # Checkpoint & distributed utilities (copy from examples/)
  requirements.txt      # Dependencies installed on TPU VMs
  spotax_setup.sh       # Pre-install script (optional)
```

### `spotax_utils.py`

SpotJAX handles infrastructure recovery automatically — on preemption it provisions a new TPU and reruns your script. But without checkpointing, your training would restart from step 0 every time. `spotax_utils.py` bridges this gap: it saves model state to GCS and restores it on retry, so training resumes from where it left off.

Copy this file from [`examples/`](examples/) into your project. It provides checkpoint management and distributed training setup with **no runtime dependency on the spotax package**.

```python
from spotax_utils import CheckpointManager, get_config, setup_distributed

config = get_config()
setup_distributed()  # Initialize JAX distributed runtime

ckpt = CheckpointManager(config.checkpoint_dir, save_interval_steps=1000)
state, start_step = ckpt.restore_or_init(initial_state)

for step in range(start_step, max_steps):
    state = train_step(state, batch)
    ckpt.save(step, state)

    if ckpt.reached_preemption(step):
        break  # Orbax already saved checkpoint, orchestrator will retry

ckpt.close()
```

**How checkpointing works:**
- SpotJAX enables GCP's [autocheckpoint](https://cloud.google.com/tpu/docs/automated-checkpointing). On preemption, GCP sends SIGTERM to the VM.
- [Orbax](https://github.com/google/orbax) catches SIGTERM and saves a checkpoint automatically, even outside `save_interval_steps`.
- `reached_preemption()` detects this across all hosts and returns `True` so your script exits cleanly.
- The orchestrator then provisions a new TPU and reruns. `restore_or_init()` picks up from the last checkpoint.

### `requirements.txt`

Standard pip requirements. SpotJAX installs them on each TPU VM using [uv](https://github.com/astral-sh/uv) with the [JAX TPU releases](https://storage.googleapis.com/jax-releases/libtpu_releases.html) index. Include `jax[tpu]` and any other dependencies your script needs:

```
jax[tpu]
flax
optax
orbax-checkpoint
grain
```

### `spotax_setup.sh` (optional)

Runs before `requirements.txt` installation. Use it for things pip can't handle: system packages, building from source, patching libraries. The venv is already activated when this runs.

## Environment Variables

SpotJAX injects these into your training script (read them via `get_config()`):

| Variable | Description |
|---|---|
| `SPOT_CHECKPOINT_DIR` | GCS path for checkpoints (`gs://bucket/job-id/ckpt`) |
| `SPOT_LOG_DIR` | GCS path for logs |
| `SPOT_JOB_ID` | Unique job identifier |
| `SPOT_IS_RESTART` | `"true"` if resuming after preemption |

JAX auto-discovers TPU topology (coordinator address, process count, process ID) from TPU metadata — no manual configuration needed for multi-node.

## CLI Reference

```bash
spotax run <script> [options]
```

| Option | Default | Description |
|---|---|---|
| `--tpu`, `-t` | `v5litepod-1` | TPU type |
| `--zone`, `-z` | `us-central1-a` | GCP zone |
| `--project`, `-p` | auto-detect | GCP project ID |
| `--bucket`, `-b` | `spotax-{project}` | GCS bucket for checkpoints |
| `--name`, `-n` | timestamp | Job name |
| `--max-retries` | `5` | Max restart attempts |
| `--stream-worker`, `-w` | `0` | Worker index to stream logs from |
| `--code-dir`, `-c` | script's parent dir | Directory to upload |

## Examples

- **[ImageNet EfficientNet](examples/imagenet_efficientnet/)** - Train EfficientNet-B2 on ImageNet-1K with ArrayRecord data pipeline
- **[MaxText Qwen3 SFT](examples/maxtext_qwen3_sft/)** - Fine-tune Qwen3 on GSM8K math problems using MaxText

## Requirements

- Python 3.10+
- GCP project with TPU API enabled and Spot TPU quota
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) authenticated with Application Default Credentials

## License

Apache 2.0
