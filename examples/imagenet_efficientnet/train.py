import argparse
import logging
import time

import jax
import jax.numpy as jnp
import optax
from bonsai.models.efficientnet.modeling import EfficientNet, ModelConfig
from bonsai.models.efficientnet.params import create_efficientnet_from_pretrained
from data import create_dataloader
from flax import nnx
from spotax_utils import CheckpointManager, get_config, setup_distributed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
log = logging.getLogger(__name__)


def create_model(num_classes: int = 1000, pretrained: bool = True) -> EfficientNet:
    """Create EfficientNet-B2 model.

    Args:
        num_classes: Number of output classes (1000 for ImageNet)
        pretrained: Load pretrained weights from timm

    Returns:
        EfficientNet model
    """
    if pretrained:
        log.info("Loading pretrained EfficientNet-B2 from timm...")
        model = create_efficientnet_from_pretrained(version=2)  # B2
        log.info("Loaded pretrained weights")
    else:
        log.info("Creating EfficientNet-B2 from scratch...")
        cfg = ModelConfig.b2()
        if num_classes != 1000:
            # Modify config for different number of classes
            cfg = ModelConfig(
                width_coefficient=cfg.width_coefficient,
                depth_coefficient=cfg.depth_coefficient,
                resolution=cfg.resolution,
                dropout_rate=cfg.dropout_rate,
                stem_conv_padding=cfg.stem_conv_padding,
                bn_momentum=cfg.bn_momentum,
                bn_epsilon=cfg.bn_epsilon,
                block_configs=cfg.block_configs,
                num_classes=num_classes,
            )
        model = EfficientNet(cfg, rngs=nnx.Rngs(0))

    # Provide RNG for dropout (Bonsai doesn't set one during init)
    model.dropout.rngs = nnx.Rngs(0)

    # Count parameters
    params = nnx.state(model, nnx.Param)
    param_count = sum(p.size for p in jax.tree.leaves(params))
    log.info(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    return model


def compute_loss(model: EfficientNet, images: jnp.ndarray, labels: jnp.ndarray):
    """Forward pass and cross-entropy loss.

    Args:
        model: EfficientNet model
        images: Batch of images [B, H, W, C]
        labels: Batch of labels [B]

    Returns:
        (loss, metrics_dict)
    """
    logits = model(images, training=True)  # [B, num_classes]

    # Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = loss.mean()

    # Accuracy
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).mean()

    # Top-5 accuracy
    top5_preds = jax.lax.top_k(logits, 5)[1]  # [B, 5]
    top5_correct = (top5_preds == labels[:, None]).any(axis=-1)
    top5_accuracy = top5_correct.mean()

    return loss, {"loss": loss, "acc": accuracy, "top5_acc": top5_accuracy}


def train(
    data_dir: str = "",
    batch_size: int = 64,
    max_steps: int = 100000,
    lr: float = 1e-3,
    warmup_steps: int = 1000,
    weight_decay: float = 1e-4,
    log_every: int = 50,
    save_every: int = 1000,
    pretrained: bool = False,
    use_synthetic: bool = False,
    shuffle: bool = False,
    prefetch_workers: int = 4,
):
    """Main training function.

    Args:
        data_dir: GCS path to ImageNet TFRecords (e.g., gs://bucket/imagenet)
        batch_size: Per-device batch size
        max_steps: Maximum training steps
        lr: Peak learning rate
        warmup_steps: Linear warmup steps
        weight_decay: AdamW weight decay
        log_every: Log metrics every N steps
        save_every: Save checkpoint every N steps
        pretrained: Start from pretrained weights
        use_synthetic: Use synthetic data for testing
        shuffle: Whether to shuffle data (default False, gcsfuse is slow with random access)
        prefetch_workers: Number of background workers for data prefetching
    """
    # Setup distributed training
    cfg = get_config()
    setup_distributed(cfg)

    ckpt_dir = cfg.checkpoint_dir
    worker_id = cfg.worker_id
    num_workers = cfg.num_workers
    num_devices = jax.device_count()

    log.info(f"Worker {worker_id}/{num_workers}, devices: {num_devices}, "
             f"platform: {jax.devices()[0].platform}, "
             f"global batch size: {batch_size * num_devices}")

    # Create model
    model = create_model(pretrained=pretrained)

    # Optimizer with warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=max_steps,
        end_value=lr * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Checkpointing
    ckpt = CheckpointManager(ckpt_dir, save_interval_steps=save_every)
    _, state = nnx.split(model)
    state, start_step = ckpt.restore_or_init(state, 0)
    if start_step > 0:
        log.info(f"Restored from step {start_step}")
        model = nnx.merge(nnx.split(model)[0], state)
        opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Data loader
    image_size = 260  # EfficientNet-B2 resolution
    loader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        is_training=True,
        worker_id=worker_id,
        num_workers=num_workers,
        use_synthetic=use_synthetic,
        shuffle=shuffle,
        prefetch_workers=prefetch_workers,
    )
    data_iter = iter(loader)

    # JIT-compiled train step with data parallelism
    @nnx.jit
    def train_step(model, opt, images, labels):
        def loss_fn(m):
            return compute_loss(m, images, labels)

        (_, metrics), grads = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
        )(model)
        opt.update(model, grads)
        return metrics

    # First step (includes JIT compilation)
    log.info(f"Starting training from step {start_step}...")
    batch = next(data_iter)
    images = jnp.array(batch["image"])
    labels = jnp.array(batch["label"])
    t0 = time.time()
    metrics = train_step(model, opt, images, labels)
    jax.block_until_ready(metrics)
    log.info(f"JIT compilation done in {time.time() - t0:.1f}s")

    step = start_step + 1
    t0 = time.time()
    running_loss = float(metrics["loss"])
    running_acc = float(metrics["acc"])
    running_top5 = float(metrics["top5_acc"])

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        images = jnp.array(batch["image"])
        labels = jnp.array(batch["label"])
        metrics = train_step(model, opt, images, labels)
        step += 1

        # Accumulate metrics
        running_loss += float(metrics["loss"])
        running_acc += float(metrics["acc"])
        running_top5 += float(metrics["top5_acc"])

        # Log
        if step % log_every == 0:
            dt = time.time() - t0
            avg_loss = running_loss / log_every
            avg_acc = running_acc / log_every
            avg_top5 = running_top5 / log_every
            steps_per_sec = log_every / dt
            images_per_sec = steps_per_sec * batch_size * num_devices

            log.info(
                f"Step {step}/{max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {avg_acc:.2%} | "
                f"Top5: {avg_top5:.2%} | "
                f"{steps_per_sec:.1f} steps/s | "
                f"{images_per_sec:.0f} img/s"
            )

            running_loss = 0.0
            running_acc = 0.0
            running_top5 = 0.0
            t0 = time.time()

        # Save checkpoint
        if step % save_every == 0:
            _, state = nnx.split(model)
            ckpt.save(step, state)
            if ckpt.reached_preemption(step):
                log.info("Preemption detected")
                break

    # Final save
    log.info(f"Training complete at step {step}")
    _, state = nnx.split(model)
    ckpt.save(step, state)
    ckpt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train EfficientNet-B2 on ImageNet")
    p.add_argument(
        "--data_dir",
        default="gs://imagenet-training-tpu ",
        help="GCS path to ImageNet ArrayRecords (expects train_arrayrecord/ subdir)",
    )
    p.add_argument("--batch_size", type=int, default=128, help="Per-device batch size")
    p.add_argument("--max_steps", type=int, default=10000, help="Max training steps")
    p.add_argument("--lr", type=float, default=1e-1, help="Peak learning rate")
    p.add_argument("--warmup_steps", type=int, default=1500, help="LR warmup steps")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--log_every", type=int, default=50, help="Log every N steps")
    p.add_argument("--save_every", type=int, default=100, help="Save every N steps")
    p.add_argument(
        "--pretrained", action="store_true", help="Start from pretrained weights (default: train from scratch)"
    )
    p.add_argument(
        "--use_synthetic", action="store_true", help="Use synthetic data for testing"
    )
    p.add_argument(
        "--shuffle", action="store_true", help="Enable data shuffling (disabled by default)"
    )
    p.add_argument(
        "--prefetch_workers",
        type=int,
        default=2,
        help="Number of prefetch workers (0=disabled, default=2)",
    )
    args = p.parse_args()

    train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        save_every=args.save_every,
        pretrained=args.pretrained,
        use_synthetic=args.use_synthetic,
        shuffle=args.shuffle,
        prefetch_workers=args.prefetch_workers,
    )
