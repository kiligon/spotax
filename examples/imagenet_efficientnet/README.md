# ImageNet EfficientNet-B2

Train [EfficientNet-B2](https://arxiv.org/abs/1905.11946) on ImageNet-1K using JAX/Flax on Spot TPUs with automatic preemption recovery.

## Data Preparation

### 1. Download ImageNet TFRecords

Download ImageNet-1K in TFRecord format from Kaggle:

- [ImageNet-1K TFRecords (Part 1)](https://www.kaggle.com/datasets/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-1)

### 2. Convert to ArrayRecord

The training pipeline uses [ArrayRecord](https://github.com/google/array_record) + [msgpack](https://msgpack.org/) instead of TFRecord to avoid a TensorFlow dependency at training time.

```bash
pip install tensorflow array-record msgpack

# Convert training set
python convert_to_arrayrecord_local.py \
    --input "path/to/train/*.tfrecord" \
    --output "path/to/train_arrayrecord/"

# Convert validation set
python convert_to_arrayrecord_local.py \
    --input "path/to/validation/*.tfrecord" \
    --output "path/to/validation_arrayrecord/"
```

### 3. Upload to GCS

```bash
gsutil -m cp -r train_arrayrecord/ gs://your-bucket/imagenet/train_arrayrecord/
gsutil -m cp -r validation_arrayrecord/ gs://your-bucket/imagenet/validation_arrayrecord/
```

## Training

```bash
# Single-host TPU
spotax run train.py --tpu v5litepod-1 \
    -- --data_dir gs://your-bucket/imagenet --max_steps 10000

# Multi-host TPU
spotax run train.py --tpu v5litepod-16 \
    -- --data_dir gs://your-bucket/imagenet --batch_size 64

# Synthetic data (no dataset needed, for testing)
spotax run train.py --tpu v5litepod-1 -- --use_synthetic --max_steps 100
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | - | GCS path to ImageNet (expects `train_arrayrecord/` subdir) |
| `--batch_size` | `128` | Per-device batch size |
| `--max_steps` | `10000` | Training steps |
| `--lr` | `2e-3` | Peak learning rate |
| `--warmup_steps` | `1500` | Linear warmup steps |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--save_every` | `1000` | Checkpoint interval |
| `--log_every` | `50` | Logging interval |
| `--pretrained` | off | Start from pretrained weights instead of training from scratch |
| `--use_synthetic` | off | Use synthetic data (no GCS needed) |
| `--shuffle` | off | Shuffle data (slower with gcsfuse) |

## How It Works

**Model:** EfficientNet-B2 (7.8M params) via [Bonsai](https://github.com/jax-ml/bonsai), with optional pretrained weights from [timm](https://github.com/huggingface/pytorch-image-models).

**Data pipeline:** GCS bucket is mounted via [gcsfuse](https://github.com/GoogleCloudPlatform/gcsfuse), ArrayRecord files are read by [Grain](https://github.com/google/grain) with msgpack deserialization, JPEG decode, random crop/flip, and ImageNet normalization.

**Optimizer:** AdamW with linear warmup + cosine decay and gradient clipping (max norm 1.0).

**Preemption recovery:** Orbax saves checkpoints every `--save_every` steps and automatically on SIGTERM. On restart, training resumes from the last checkpoint via `restore_or_init()`.

## Project Files

| File | Purpose |
|---|---|
| `train.py` | Training script |
| `data.py` | Grain data pipeline with gcsfuse GCS access |
| `spotax_utils.py` | Checkpoint management and distributed setup |
| `spotax_setup.sh` | Installs gcsfuse and bonsai on TPU VMs |
| `requirements.txt` | Python dependencies |
| `convert_to_arrayrecord.py` | TFRecord to ArrayRecord conversion (GCS) |
| `convert_to_arrayrecord_local.py` | TFRecord to ArrayRecord conversion (local) |
