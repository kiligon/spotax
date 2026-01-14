# GSM8K Math SFT Example

A minimal example of supervised fine-tuning (SFT) using SpotJAX on TPUs.

## What This Does

Fine-tunes Qwen3-0.6B on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) grade school math:
- Input: `"Natalia sold clips to 48 of her friends in April..."`
- Output: Step-by-step reasoning ending with `#### <answer>`

Demonstrates the full SpotJAX pipeline with real data:
- HuggingFace datasets loading (cached locally)
- Grain data loading with distributed sharding
- Bonsai Qwen3 model (JAX/Flax NNX)
- Orbax checkpointing with preemption recovery
- Multi-host TPU training

## Files

```
simple_math_sft/
├── train.py          # Main training script
├── data.py           # Grain data loader + math problem generator
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Quick Start

### 1. Local Testing (CPU/GPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Run training locally (100 steps)
# First run downloads GSM8K dataset (~10MB)
python train.py --local --max_steps=100 --batch_size=4
```

### 2. Single-Host TPU (v5e-8)

```bash
# Navigate to example directory
cd examples/simple_math_sft

# Generate SpotJAX utilities
spotjax init

# Run on spot TPU
spotjax run \
  --tpu=v5e-8 \
  --zone=us-central1-a \
  --script=train.py \
  -- --max_steps=500 --batch_size=16
```

### 3. Multi-Host TPU (v5e-32)

```bash
spotjax run \
  --tpu=v5e-32 \
  --zone=us-central1-a \
  --script=train.py \
  -- --max_steps=1000 --batch_size=8
```

With 4 hosts (v5e-32), each host gets a shard of the data automatically.

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen3-0.6B` | HuggingFace model name |
| `--batch_size` | 8 | Batch size per device |
| `--max_length` | 512 | Maximum sequence length |
| `--max_steps` | 1000 | Training steps |
| `--lr` | 1e-5 | Learning rate |
| `--log_every` | 10 | Log every N steps |
| `--save_every` | 100 | Checkpoint every N steps |
| `--local` | False | Run without SpotJAX |

## How It Works

### Data Pipeline (HuggingFace + Grain)

```python
# GSM8K loaded from HuggingFace (~7.5K examples, ~10MB)
# Cached to ~/.cache/huggingface/datasets (survives preemption if VM persists)

# Example format:
"<|im_start|>user\nNatalia sold clips to 48 of her friends...<|im_end|>\n"
"<|im_start|>assistant\nNatalia sold 48/2 = 24 clips in May...\n#### 72<|im_end|>"

# Grain handles:
# - Tokenization (max_length=512 for longer GSM8K answers)
# - Batching
# - Distributed sharding (each worker gets different data)
```

**Data strategy for larger datasets**: For production with GB+ datasets, pre-stage to
GCS and use ArrayRecord format for optimal streaming. GSM8K is small enough for
HuggingFace's default caching.

### Model (Bonsai Qwen3)

Uses [jax-ml/bonsai](https://github.com/jax-ml/bonsai) Qwen3 implementation:
- ~300 lines of clean JAX code
- Flax NNX API
- Loads weights from HuggingFace

### Checkpointing (SpotJAX + Orbax)

```python
from spotjax_utils import CheckpointManager, get_config

config = get_config()
ckpt = CheckpointManager(config.checkpoint_dir)

# Restore if resuming after preemption
state, start_step = ckpt.restore_or_init(initial_state)

for step in range(start_step, max_steps):
    state = train_step(state, batch)
    ckpt.save(step, state)

    # Exit gracefully on preemption
    if ckpt.reached_preemption(step):
        break
```

## Preemption Recovery

When a Spot TPU is preempted:
1. GCP sends SIGTERM to the VM
2. Orbax automatically saves checkpoint
3. `reached_preemption()` returns True
4. Training exits gracefully
5. SpotJAX provisions a new TPU
6. Training resumes from checkpoint

## Expected Results

On v5e-8 with default settings:
- First run downloads GSM8K (~10MB) + model weights (~1GB)
- Training: ~10-15 min for 1000 steps
- Loss should drop from ~2.0 to ~0.5
- Accuracy should reach ~70-80% (GSM8K is harder than synthetic math)

## Scaling Up

To use a larger model:

```bash
# Qwen3-1.7B (needs more memory)
spotjax run --tpu=v5e-16 --script=train.py -- --model=Qwen/Qwen3-1.7B --batch_size=4

# Qwen3-4B
spotjax run --tpu=v5e-32 --script=train.py -- --model=Qwen/Qwen3-4B --batch_size=2
```

## Troubleshooting

**OOM errors**: Reduce `--batch_size` or `--max_length`, or use larger TPU

**Slow data loading**: GSM8K is cached after first download. For larger datasets,
consider pre-staging to GCS.

**Model download slow**: First run downloads ~1GB weights from HuggingFace

**Preemption loops**: Check `SPOT_IS_RESTART` handling in train.py
