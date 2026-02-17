"""MaxText Qwen3 SFT on GSM8K, orchestrated by SpotJAX.

1. Auto-converts HuggingFace checkpoint to MaxText format if needed
2. Runs MaxText SFT trainer via Python API (Orbax handles SIGTERM for preemption)

Usage:
    spotax run train.py --tpu v5litepod-8 --zone us-central1-a

Edit config.py to change model, dataset, or training parameters.
"""

import logging
import os
import subprocess
import sys

from config import SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
log = logging.getLogger(__name__)

MAXTEXT_DIR = "$HOME/maxtext"


def ensure_torch():
    """Install PyTorch CPU if not available (needed for HF->MaxText conversion)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        log.info("PyTorch not found, installing CPU-only version...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torch",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
        )


def gcs_path_exists(gcs_path: str) -> bool:
    """Check if a GCS path exists."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def ensure_converted(maxtext_dir: str, base_output_dir: str, model_name: str) -> str:
    """Convert HuggingFace checkpoint to MaxText format if not already done.

    Returns:
        GCS path to the converted checkpoint.
    """
    converted_path = f"{base_output_dir}/converted/0/items"

    if gcs_path_exists(converted_path):
        log.info(f"Converted checkpoint exists: {converted_path}")
        return converted_path

    ensure_torch()

    log.info(f"Converting {model_name} from HuggingFace to MaxText format...")
    base_config = os.path.join(maxtext_dir, "src/maxtext/configs/base.yml")

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "MaxText.utils.ckpt_conversion.to_maxtext",
            base_config,
            f"model_name={model_name}",
            f"base_output_directory={base_output_dir}/converted",
            "scan_layers=true",
            "skip_jax_distributed_system=True",
        ],
        env=env,
    )

    if result.returncode != 0:
        log.error("Checkpoint conversion failed")
        sys.exit(1)

    log.info(f"Conversion complete: {converted_path}")
    return converted_path


def main():
    checkpoint_dir = os.environ.get("SPOT_CHECKPOINT_DIR", "")
    if not checkpoint_dir:
        log.error("SPOT_CHECKPOINT_DIR not set. Run with: spotax run train.py ...")
        sys.exit(1)

    cfg = SFTConfig()
    maxtext_dir = os.path.expandvars(MAXTEXT_DIR)
    sft_config = os.path.join(maxtext_dir, "src/maxtext/configs/post_train/sft.yml")
    chat_template = os.path.join(maxtext_dir, cfg.chat_template)

    # Convert HF checkpoint to MaxText format (skips if already done)
    load_path = ensure_converted(maxtext_dir, checkpoint_dir, cfg.model_name)

    # Initialize MaxText config via pyconfig (same as CLI argv)
    from MaxText import pyconfig

    config = pyconfig.initialize([
        "",  # argv[0] placeholder
        sft_config,
        "run_name=qwen3_sft",
        f"base_output_directory={checkpoint_dir}",
        f"model_name={cfg.model_name}",
        f"load_parameters_path={load_path}",
        f"tokenizer_path={cfg.tokenizer_path}",
        f"hf_path={cfg.hf_path}",
        f"hf_data_dir={cfg.hf_data_dir}",
        f"train_split={cfg.train_split}",
        f"train_data_columns={cfg.train_data_columns}",
        f"chat_template_path={chat_template}",
        f"steps={cfg.steps}",
        f"per_device_batch_size={cfg.per_device_batch_size}",
        f"max_target_length={cfg.max_target_length}",
        f"learning_rate={cfg.learning_rate}",
        f"weight_dtype={cfg.weight_dtype}",
        f"dtype={cfg.dtype}",
        f"checkpoint_period={cfg.checkpoint_period}",
        f"async_checkpointing={str(cfg.async_checkpointing).lower()}",
        "dataset_type=hf",
    ])

    log.info(f"Starting MaxText SFT: {cfg.model_name} on {cfg.hf_path}")

    # Run training via Python API
    # Orbax registers SIGTERM handler internally for preemption-safe checkpointing
    from maxtext.trainers.post_train.sft import train_sft

    train_sft.train(config)

    log.info("Training complete.")
    # Force exit — Grain multiprocessing workers and async checkpoint threads
    # can hang indefinitely after training finishes
    os._exit(0)


if __name__ == "__main__":
    main()
