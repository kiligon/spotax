"""ImageNet data loading with Grain and JAX.

Expects ImageNet in TFRecord format at:
    gs://your-bucket/imagenet/train/*.tfrecord
    gs://your-bucket/imagenet/validation/*.tfrecord

Each TFRecord contains examples with:
    - 'image/encoded': JPEG bytes
    - 'image/class/label': int64 label (0-999)

Uses array_record for efficient random access on GCS.
"""

import io
import logging
from functools import partial

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ImageNet mean/std (RGB order)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SimpleImageSource(grain.RandomAccessDataSource):
    """Simple in-memory data source for testing (synthetic data)."""

    def __init__(
        self, num_examples: int = 10000, image_size: int = 260, num_classes: int = 1000
    ):
        self.num_examples = num_examples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> dict:
        # Generate deterministic synthetic data based on index
        rng = np.random.default_rng(idx)
        image = rng.standard_normal((self.image_size, self.image_size, 3)).astype(
            np.float32
        )
        # Apply normalization like real data
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        label = idx % self.num_classes
        return {"image": image, "label": label}


class TFRecordSource(grain.RandomAccessDataSource):
    """Data source for TFRecord files on GCS using tf.data for parsing only."""

    def __init__(self, tfrecord_pattern: str, shuffle_files: bool = True):
        """Initialize TFRecord data source.

        Args:
            tfrecord_pattern: Glob pattern for TFRecord files (e.g., gs://bucket/train/*.tfrecord)
            shuffle_files: Whether to shuffle file order
        """
        import tensorflow as tf

        self._files = tf.io.gfile.glob(tfrecord_pattern)
        if not self._files:
            raise ValueError(f"No TFRecord files found: {tfrecord_pattern}")

        if shuffle_files:
            np.random.shuffle(self._files)

        log.info(f"Found {len(self._files)} TFRecord files")

        # Build index: map global index -> (file_idx, local_idx)
        self._index = []
        self._file_data = []  # Store parsed data per file

        for file_idx, path in enumerate(self._files):
            ds = tf.data.TFRecordDataset(path)
            count = sum(1 for _ in ds)
            for local_idx in range(count):
                self._index.append((file_idx, local_idx))
            self._file_data.append(None)  # Lazy load

        log.info(f"Total examples: {len(self._index)}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        import tensorflow as tf

        file_idx, local_idx = self._index[idx]

        # Lazy load and cache file data
        if self._file_data[file_idx] is None:
            ds = tf.data.TFRecordDataset(self._files[file_idx])
            parsed = []
            for raw in ds:
                features = tf.io.parse_single_example(
                    raw,
                    features={
                        "image/encoded": tf.io.FixedLenFeature([], tf.string),
                        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
                    },
                )
                parsed.append(
                    {
                        "image_bytes": features["image/encoded"].numpy(),
                        "label": int(features["image/class/label"].numpy()),
                    }
                )
            self._file_data[file_idx] = parsed

        return self._file_data[file_idx][local_idx]


class DecodeOp(grain.MapTransform):
    """Decode JPEG image bytes to numpy array using PIL."""

    def map(self, example: dict) -> dict:
        image_bytes = example["image_bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0
        return {"image": image, "label": example["label"]}


class RandomResizedCropOp(grain.MapTransform):
    """Random resized crop for training augmentation."""

    def __init__(self, size: int, scale: tuple = (0.08, 1.0), ratio: tuple = (3 / 4, 4 / 3)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def map(self, example: dict) -> dict:
        image = example["image"]
        h, w = image.shape[:2]

        # Random scale and aspect ratio
        rng = np.random.default_rng()
        for _ in range(10):  # Try up to 10 times
            area = h * w
            target_area = rng.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = rng.uniform(self.ratio[0], self.ratio[1])

            new_w = int(np.sqrt(target_area * aspect_ratio))
            new_h = int(np.sqrt(target_area / aspect_ratio))

            if 0 < new_w <= w and 0 < new_h <= h:
                top = rng.integers(0, h - new_h + 1)
                left = rng.integers(0, w - new_w + 1)

                # Crop
                image = image[top : top + new_h, left : left + new_w]
                break
        else:
            # Fallback: center crop
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            image = image[top : top + min_dim, left : left + min_dim]

        # Resize using PIL
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize((self.size, self.size), Image.BILINEAR)
        image = np.array(pil_image, dtype=np.float32) / 255.0

        return {"image": image, "label": example["label"]}


class CenterCropOp(grain.MapTransform):
    """Center crop and resize for evaluation."""

    def __init__(self, size: int, crop_ratio: float = 0.875):
        self.size = size
        self.crop_ratio = crop_ratio

    def map(self, example: dict) -> dict:
        image = example["image"]
        h, w = image.shape[:2]

        # Resize so shorter side is size/crop_ratio
        target_short = int(self.size / self.crop_ratio)
        if h < w:
            new_h = target_short
            new_w = int(w * target_short / h)
        else:
            new_w = target_short
            new_h = int(h * target_short / w)

        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)
        image = np.array(pil_image, dtype=np.float32) / 255.0

        # Center crop
        h, w = image.shape[:2]
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        image = image[top : top + self.size, left : left + self.size]

        return {"image": image, "label": example["label"]}


class RandomHorizontalFlipOp(grain.RandomMapTransform):
    """Random horizontal flip."""

    def random_map(self, example: dict, rng: np.random.Generator) -> dict:
        if rng.random() < 0.5:
            example["image"] = np.fliplr(example["image"]).copy()
        return example


class NormalizeOp(grain.MapTransform):
    """Apply ImageNet normalization."""

    def map(self, example: dict) -> dict:
        image = example["image"]
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return {"image": image.astype(np.float32), "label": example["label"]}


def create_dataloader(
    data_dir: str,
    batch_size: int = 64,
    image_size: int = 260,
    is_training: bool = True,
    seed: int = 42,
    worker_id: int = 0,
    num_workers: int = 1,
    use_synthetic: bool = False,
) -> grain.DataLoader:
    """Create Grain dataloader for ImageNet.

    Args:
        data_dir: GCS path containing train/ and validation/ subdirs with TFRecords
        batch_size: Per-device batch size
        image_size: Target image size (260 for EfficientNet-B2)
        is_training: Training or evaluation mode
        seed: Random seed
        worker_id: Current worker index (for multi-node)
        num_workers: Total number of workers
        use_synthetic: Use synthetic data for testing (no GCS needed)

    Returns:
        Grain DataLoader yielding batches of {"image": [...], "label": [...]}
    """
    if use_synthetic:
        log.info("Using synthetic data for testing")
        source = SimpleImageSource(
            num_examples=50000 if is_training else 10000,
            image_size=image_size,
            num_classes=1000,
        )
        operations = [grain.Batch(batch_size, drop_remainder=True)]
    else:
        split = "train" if is_training else "validation"
        pattern = f"{data_dir}/{split}/*.tfrecord"
        log.info(f"Loading ImageNet from {pattern}")

        source = TFRecordSource(pattern, shuffle_files=is_training)

        if is_training:
            operations = [
                DecodeOp(),
                RandomResizedCropOp(size=image_size),
                RandomHorizontalFlipOp(),
                NormalizeOp(),
                grain.Batch(batch_size, drop_remainder=True),
            ]
        else:
            operations = [
                DecodeOp(),
                CenterCropOp(size=image_size),
                NormalizeOp(),
                grain.Batch(batch_size, drop_remainder=True),
            ]

    return grain.DataLoader(
        data_source=source,
        sampler=grain.IndexSampler(
            num_records=len(source),
            num_epochs=None,  # Infinite iteration
            seed=seed,
            shuffle=is_training,
            shard_options=grain.ShardOptions(
                worker_id, num_workers, drop_remainder=True
            ),
        ),
        operations=operations,
    )
