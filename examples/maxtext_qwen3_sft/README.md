# MaxText Qwen3 SFT on GSM8K

Fine-tunes Qwen3 on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) math problems using [MaxText](https://github.com/AI-Hypercomputer/maxtext) — Google's high-performance JAX training framework for TPUs.

## Quick Start

```bash
# 1. Edit config.py to set model size, dataset, training params
#    Defaults: Qwen3-0.6B, GSM8K, 500 steps

# 2. Run on a Spot TPU
spotax run examples/maxtext_qwen3_sft/train.py \
    --tpu v5litepod-8 \
    --zone us-central1-a
```

That's it. SpotJAX handles TPU provisioning, MaxText installation, checkpoint conversion, training, and preemption recovery.

## What Happens Under the Hood

```
spotax run train.py
    │
    ├── Provision Spot TPU
    ├── Upload code via rsync
    ├── Run spotax_setup.sh
    │   ├── Clone MaxText
    │   ├── Install MaxText + tunix + PyTorch CPU
    │   └── ~5-10 min setup time
    │
    └── Run train.py
        ├── Auto-convert HF checkpoint → MaxText format (first run only)
        └── os.execvp → MaxText SFT trainer
            ├── Training with automatic checkpointing to GCS
            └── On preemption: Orbax auto-saves → SpotJAX retries
```

## Configuration

Edit `config.py` before running. All fields:

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `qwen3-0.6b` | MaxText model name (`qwen3-0.6b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen3-32b`) |
| `tokenizer_path` | `Qwen/Qwen3-0.6B` | HuggingFace tokenizer ID (must match model) |
| `hf_path` | `openai/gsm8k` | HuggingFace dataset |
| `train_data_columns` | `['question','answer']` | Dataset columns to use |
| `chat_template` | `...chat_templates/math_qa.json` | How to format prompt/completion |
| `steps` | `500` | Training steps |
| `per_device_batch_size` | `1` | Batch size per TPU chip |
| `max_target_length` | `1024` | Max sequence length |
| `learning_rate` | `3e-6` | Learning rate |
| `checkpoint_period` | `100` | Save checkpoint every N steps |

## Scaling to Larger Models

Change `model_name` and `tokenizer_path` in `config.py`, and use a larger TPU:

| Model | TPU | config.py changes |
|-------|-----|-------------------|
| Qwen3-0.6B | v5litepod-8 | Default |
| Qwen3-4B | v5litepod-8 | `model_name="qwen3-4b"`, `tokenizer_path="Qwen/Qwen3-4B"` |
| Qwen3-8B | v5litepod-16 | `model_name="qwen3-8b"`, `tokenizer_path="Qwen/Qwen3-8B"` |

## Preemption Recovery

MaxText handles preemption automatically:

1. GCP sends SIGTERM → MaxText's Orbax auto-saves a checkpoint
2. SpotJAX detects exit, provisions a new TPU, re-runs setup + train.py
3. MaxText finds existing checkpoints at the same GCS path → resumes from last step

The `run_name` and `base_output_directory` are deterministic (derived from `SPOT_JOB_ID`), so MaxText always finds its checkpoints after a retry.

## Checkpoint Conversion

HuggingFace model weights must be converted to MaxText's Orbax format before training. This happens **automatically** in `train.py`:

- First run: converts and saves to `gs://bucket/job-id/ckpt/converted/`
- Subsequent runs (including retries): skips conversion, uses existing checkpoint

Conversion takes ~5 minutes for Qwen3-0.6B and uses PyTorch CPU (installed by `spotax_setup.sh`).

## Comparison with simple_math_sft Example

| | simple_math_sft | maxtext_qwen3_sft |
|---|---|---|
| Framework | Custom loop (Bonsai + Flax NNX) | MaxText |
| Lines of training code | ~180 | ~30 (wrapper only) |
| Checkpointing | Manual (spotax_utils.py) | MaxText built-in |
| Data loading | Grain (custom) | MaxText built-in (HF datasets) |
| Distributed init | `jax.distributed.initialize()` | MaxText handles it |
| Model support | Qwen3 only (via Bonsai) | Many models (Qwen3, Llama, Gemma, DeepSeek, ...) |
| Setup complexity | Bonsai patches in setup script | Clone + install MaxText |

Use **simple_math_sft** to learn how JAX training works from scratch. Use **maxtext_qwen3_sft** for production-oriented training with MaxText's optimizations.
