# SpotJAX

Run JAX training on Google Cloud Spot TPUs with automatic preemption recovery.

## Installation

```bash
pip install spotax
```

## Quick Start

```bash
# Check prerequisites
spotax setup
spotax setup --fix  # Auto-fix issues

# Run training on a Spot TPU
spotax run train.py --tpu v4-8 --zone us-central2-b
```


## Using spotax_utils.py

Copy `spotax_utils.py` from the examples folder into your project:

```bash
cp examples/simple_math_sft/spotax_utils.py ./my_project/
```

Then use it in your training script:

```python
from spotax_utils import CheckpointManager, get_config, setup_distributed

# Load config from environment (set by spotax CLI)
config = get_config()

# Initialize distributed training (required for v4-16+)
setup_distributed(config)

# Setup checkpointing
ckpt = CheckpointManager(config.checkpoint_dir)
state, start_step = ckpt.restore_or_init(initial_state)

# Training loop
for step in range(start_step, max_steps):
    state = train_step(state, batch)
    ckpt.save(step, state)

    # Exit gracefully on preemption
    if ckpt.reached_preemption(step):
        break

ckpt.close()
```

The file has no runtime dependency on spotax - it's yours to modify.

## How It Works

1. Creates GCS bucket for checkpoints
2. Provisions Spot TPU via Queued Resources API
3. SSH to all nodes, upload code via rsync
4. Install `requirements.txt` if present
5. Run script on all nodes
6. On preemption: cleanup and retry

## Requirements

- Python 3.10+
- GCP project with TPU quota
- `gcloud` CLI ([install](https://cloud.google.com/sdk/docs/install))

Run `spotax setup` to verify prerequisites.
