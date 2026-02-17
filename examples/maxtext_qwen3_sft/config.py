"""Training configuration for MaxText Qwen3 SFT.

Edit this file to change model, dataset, or training parameters.
"""

from dataclasses import dataclass


@dataclass
class SFTConfig:
    """MaxText SFT training configuration."""

    # Model
    model_name: str = "qwen3-0.6b"
    tokenizer_path: str = "Qwen/Qwen3-0.6B"

    # Dataset
    hf_path: str = "openai/gsm8k"
    hf_data_dir: str = "main"
    train_split: str = "train"
    train_data_columns: str = "['question','answer']"
    chat_template: str = "src/maxtext/examples/chat_templates/math_qa.json"

    # Training
    steps: int = 500
    per_device_batch_size: int = 1
    max_target_length: int = 1024
    learning_rate: float = 3e-6
    weight_dtype: str = "bfloat16"
    dtype: str = "bfloat16"

    # Checkpointing
    checkpoint_period: int = 100
    async_checkpointing: bool = True
