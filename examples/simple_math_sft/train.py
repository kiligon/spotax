"""GSM8K SFT training with Bonsai Qwen3.

Fine-tunes Qwen3 on GSM8K grade school math word problems.

Usage:
    python train.py --local --max_steps=100
    spotjax run --tpu=v5e-8 --script=train.py
"""

import argparse
import logging
import os
import time

import jax.numpy as jnp
import optax
from flax import nnx
from huggingface_hub import snapshot_download

from bonsai.models.qwen3.modeling import Qwen3, ModelConfig
from bonsai.models.qwen3.params import create_model_from_safe_tensors
from data import create_dataloader, get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
log = logging.getLogger(__name__)


def load_model(model_name: str) -> tuple[Qwen3, ModelConfig, int]:
    """Load Qwen3 model from HuggingFace."""
    log.info(f"Loading {model_name}...")
    path = snapshot_download(model_name)

    configs = {
        "0.6B": ModelConfig.qwen3_0_6b,
        "1.7B": ModelConfig.qwen3_1_7b,
        "4B": ModelConfig.qwen3_4b,
        "8B": ModelConfig.qwen3_8b,
    }
    config_fn = next((v for k, v in configs.items() if k in model_name), ModelConfig.qwen3_0_6b)
    config = config_fn(use_sharding=False)

    model = create_model_from_safe_tensors(path, config)
    pad_id = get_tokenizer(model_name).pad_token_id
    log.info(f"Loaded: {config.num_layers} layers, {config.emb_dim} dim")
    return model, config, pad_id


def compute_loss(model: Qwen3, input_ids, labels, pad_id: int, config: ModelConfig):
    """Forward pass and cross-entropy loss."""
    batch_size, seq_len = input_ids.shape
    segment_ids = (input_ids != pad_id).astype(jnp.int32)
    cache = model.init_cache(config, batch_size, seq_len, generate_steps=0)
    logits = model(input_ids, segment_ids, cache, num_right_pads=0)

    # Next-token prediction loss
    logits, labels = logits[:, :-1], labels[:, 1:]
    mask = labels != -100
    labels = jnp.where(mask, labels, 0)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
    )
    loss = jnp.where(mask.reshape(-1), loss, 0.0).sum() / mask.sum()
    acc = ((logits.argmax(-1) == labels) & mask).sum() / mask.sum()
    return loss, {"loss": loss, "acc": acc}


def train(
    model_name: str = "Qwen/Qwen3-0.6B",
    batch_size: int = 8,
    max_length: int = 512,
    max_steps: int = 1000,
    lr: float = 1e-5,
    log_every: int = 10,
    save_every: int = 100,
    local: bool = False,
):
    # Setup
    if local:
        ckpt_dir = "/tmp/simple_math_sft_ckpt"
        os.makedirs(ckpt_dir, exist_ok=True)
        worker_id, num_workers, is_restart = 0, 1, False
    else:
        from spotjax_utils import get_config, setup_distributed, CheckpointManager
        cfg = get_config()
        setup_distributed(cfg)
        ckpt_dir = cfg.checkpoint_dir
        worker_id, num_workers, is_restart = cfg.worker_id, cfg.num_workers, cfg.is_restart

    log.info(f"Worker {worker_id}/{num_workers}")

    # Model & optimizer
    model, config, pad_id = load_model(model_name)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Checkpointing
    start_step = 0
    if local:
        import orbax.checkpoint as ocp
        ckpt = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=3))
        if ckpt.latest_step():
            log.info(f"Restoring from step {ckpt.latest_step()}")
            gdef, state = nnx.split(model)
            state = ckpt.restore(ckpt.latest_step(), args=ocp.args.StandardRestore(state))
            model = nnx.merge(gdef, state)
            opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
            start_step = ckpt.latest_step()
    else:
        ckpt = CheckpointManager(ckpt_dir, save_interval_steps=save_every)
        _, state = nnx.split(model)
        state, start_step = ckpt.restore_or_init(state, 0)
        if start_step > 0:
            model = nnx.merge(nnx.split(model)[0], state)
            opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Data (GSM8K loaded from HuggingFace, cached locally)
    loader = create_dataloader(batch_size, max_length, model_name, worker_id=worker_id, num_workers=num_workers)
    data = iter(loader)

    # Train step (JIT compiled)
    @nnx.jit
    def train_step(model, opt, input_ids, labels):
        def loss_fn(m):
            return compute_loss(m, input_ids, labels, pad_id, config)
        (_, metrics), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)(model)
        opt.update(model, grads)
        return metrics

    # Training loop
    log.info(f"Training from step {start_step}")
    step = start_step
    t0 = time.time()

    while step < max_steps:
        try:
            batch = next(data)
        except StopIteration:
            data = iter(loader)
            batch = next(data)

        metrics = train_step(model, opt, jnp.array(batch["input_ids"]), jnp.array(batch["labels"]))
        step += 1

        if step % log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            log.info(f"Step {step}/{max_steps} | Loss: {float(metrics['loss']):.4f} | Acc: {float(metrics['acc']):.2%} | {dt:.1f}s/{log_every} steps")

        if step % save_every == 0:
            _, state = nnx.split(model)
            if local:
                ckpt.save(step, args=ocp.args.StandardSave(state))
            else:
                ckpt.save(step, state)
                if ckpt.reached_preemption(step):
                    log.info("Preemption detected")
                    break

    log.info(f"Done at step {step}")
    _, state = nnx.split(model)
    if local:
        ckpt.save(step, args=ocp.args.StandardSave(state))
        ckpt.wait_until_finished()
    else:
        ckpt.save(step, state)
    ckpt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--local", action="store_true")
    args = p.parse_args()
    train(**vars(args))
