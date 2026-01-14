"""GSM8K dataset with Grain data loading for math SFT."""

import logging
import grain.python as grain
import numpy as np
from transformers import AutoTokenizer

log = logging.getLogger(__name__)


def load_gsm8k(split: str = "train") -> list[str]:
    """Load GSM8K and format for chat SFT.

    GSM8K contains grade school math word problems with step-by-step solutions.
    ~7.5K training examples, ~10MB download.

    HuggingFace datasets caches to ~/.cache/huggingface/datasets by default,
    so subsequent loads are instant (survives preemption if VM persists).
    """
    from datasets import load_dataset

    log.info(f"Loading GSM8K {split} split...")
    ds = load_dataset("openai/gsm8k", "main", split=split)

    texts = []
    for ex in ds:
        # GSM8K format: question + step-by-step answer ending with "#### <number>"
        text = (
            f"<|im_start|>user\n{ex['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['answer']}<|im_end|>"
        )
        texts.append(text)

    log.info(f"Loaded {len(texts)} examples")
    return texts


class TextDataSource(grain.RandomAccessDataSource):
    """Random access data source for text data."""
    def __init__(self, texts: list[str]):
        self._texts = texts

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx):
        return self._texts[idx]


class TokenizeTransform(grain.MapTransform):
    """Tokenize text for causal LM training."""

    def __init__(self, tokenizer_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def map(self, text: str) -> dict:
        enc = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="np")
        input_ids = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        labels = np.where(mask == 1, input_ids, -100)
        return {"input_ids": input_ids, "labels": labels}


def create_dataloader(
    batch_size: int = 8,
    max_length: int = 512,
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    seed: int = 42,
    worker_id: int = 0,
    num_workers: int = 1,
) -> grain.DataLoader:
    """Create Grain dataloader for GSM8K.

    Data is loaded from HuggingFace and cached locally. For larger datasets,
    consider pre-staging to GCS and using ArrayRecord format.
    """
    texts = load_gsm8k("train")

    return grain.DataLoader(
        data_source=TextDataSource(texts),
        sampler=grain.IndexSampler(
            num_records=len(texts),
            num_epochs=None,
            seed=seed,
            shuffle=True,
            shard_options=grain.ShardOptions(worker_id, num_workers, drop_remainder=True),
        ),
        operations=[
            TokenizeTransform(tokenizer_name, max_length),
            grain.Batch(batch_size, drop_remainder=True),
        ],
    )


def get_tokenizer(name: str = "Qwen/Qwen3-0.6B"):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
