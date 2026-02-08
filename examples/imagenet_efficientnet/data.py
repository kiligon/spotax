"""ImageNet data loading with Grain and ArrayRecord.

Expects ImageNet in ArrayRecord format (converted from TFRecords):
    gs://your-bucket/imagenet/train_arrayrecord/*.arrayrecord
    gs://your-bucket/imagenet/validation_arrayrecord/*.arrayrecord

Each ArrayRecord contains msgpack-serialized examples:
    {"image": <jpeg_bytes>, "label": <int>}

Uses gcsfuse to mount GCS buckets for efficient access.
See convert_to_arrayrecord.py for conversion instructions.
"""

import io
import logging
import os
import subprocess

import grain.python as grain
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ImageNet mean/std (RGB order)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_gcs_accessible(gcs_path: str) -> str:
    """Mount GCS path via gcsfuse and return local path.

    Args:
        gcs_path: GCS path like gs://bucket/path or local path

    Returns:
        Local path (either original or mounted)

    Raises:
        RuntimeError: If gcsfuse is not installed or mount fails
    """
    # Strip whitespace to handle accidental trailing spaces
    gcs_path = gcs_path.strip()

    if not gcs_path.startswith("gs://"):
        return gcs_path  # Already local

    # Parse bucket and subpath
    path_without_prefix = gcs_path[5:]  # Remove "gs://"
    parts = path_without_prefix.split("/", 1)
    bucket = parts[0]
    subpath = parts[1] if len(parts) > 1 else ""

    # Use user-writable cache directory
    cache_root = os.path.expanduser("~/.cache/gcs")
    mount_point = os.path.join(cache_root, bucket)
    local_path = os.path.join(mount_point, subpath) if subpath else mount_point

    # Check if already mounted
    if os.path.ismount(mount_point):
        log.info(f"GCS bucket already mounted: {mount_point}")
        return local_path

    # Check gcsfuse is available
    import shutil
    if not shutil.which("gcsfuse"):
        raise RuntimeError(
            "gcsfuse is not installed. Install it with:\n"
            "  sudo apt-get install gcsfuse\n"
            "Or on TPU VMs, it should be pre-installed."
        )

    # Create mount point and mount
    os.makedirs(mount_point, exist_ok=True)
    log.info(f"Mounting gs://{bucket} to {mount_point}")

    try:
        subprocess.run(
            [
                "gcsfuse",
                "--implicit-dirs",
                "--type-cache-max-size-mb=-1",
                "--stat-cache-max-size-mb=-1",
                "--kernel-list-cache-ttl-secs=-1",
                "--metadata-cache-ttl-secs=-1",
                "--max-conns-per-host=100",
                bucket,
                mount_point,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info(f"Mounted gs://{bucket} successfully")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gcsfuse mount failed: {e.stderr}") from e

    return local_path


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


class ParseMsgpackOp(grain.MapTransform):
    """Parse msgpack-serialized ArrayRecord entries."""

    def __init__(self, label_offset: int = 0):
        """Initialize parser.

        Args:
            label_offset: Offset to subtract from labels.
                         Use -1 if labels are 1-indexed (1-1000) to convert to 0-indexed (0-999).
        """
        self.label_offset = label_offset

    def map(self, raw_bytes: bytes) -> dict:
        import msgpack

        data = msgpack.unpackb(raw_bytes, raw=False)
        label = data["label"] + self.label_offset
        return {"image_bytes": data["image"], "label": label}


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
    labels_one_indexed: bool = True,
    shuffle: bool = False,  # Disabled by default (gcsfuse random access is slow)
    prefetch_workers: int = 2,  # Number of prefetch workers (0 = no prefetch)
) -> grain.DataLoader:
    """Create Grain dataloader for ImageNet.

    Args:
        data_dir: Path to ArrayRecord files (GCS or local).
                  For GCS paths (gs://...), the bucket will be mounted via gcsfuse.
                  Expected structure: {data_dir}/train_arrayrecord/ and validation_arrayrecord/
        batch_size: Per-device batch size
        image_size: Target image size (260 for EfficientNet-B2)
        is_training: Training or evaluation mode
        seed: Random seed
        worker_id: Current worker index (for multi-node)
        num_workers: Total number of workers
        use_synthetic: Use synthetic data for testing (no GCS needed)
        labels_one_indexed: If True, labels are 1-1000 and will be converted to 0-999.
                           Most ImageNet TFRecords use 1-indexed labels.
        shuffle: Whether to shuffle data. Default False (gcsfuse random access is slow).
        prefetch_workers: Number of background workers for prefetching (0 = disabled).

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
        # Mount GCS if needed
        local_data_dir = ensure_gcs_accessible(data_dir)

        # Select split
        split_name = "train_arrayrecord" if is_training else "validation_arrayrecord"
        split_path = os.path.join(local_data_dir, split_name)

        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"ArrayRecord directory not found: {split_path}\n"
                f"Run convert_to_arrayrecord.py to convert TFRecords first."
            )

        log.info(f"Loading ImageNet from {split_path}")

        # Find all ArrayRecord files in the directory
        arrayrecord_files = sorted(
            [
                os.path.join(split_path, f)
                for f in os.listdir(split_path)
                if f.endswith(".arrayrecord")
            ]
        )
        if not arrayrecord_files:
            raise FileNotFoundError(
                f"No .arrayrecord files found in {split_path}\n"
                f"Run convert_to_arrayrecord.py to convert TFRecords first."
            )
        log.info(f"Found {len(arrayrecord_files)} ArrayRecord files")

        # Use Grain's built-in ArrayRecordDataSource with file list
        source = grain.ArrayRecordDataSource(arrayrecord_files)
        log.info(f"Total examples: {len(source)}")

        # Convert 1-indexed labels (1-1000) to 0-indexed (0-999) if needed
        label_offset = -1 if labels_one_indexed else 0

        if is_training:
            operations = [
                ParseMsgpackOp(label_offset=label_offset),
                DecodeOp(),
                RandomResizedCropOp(size=image_size),
                RandomHorizontalFlipOp(),
                NormalizeOp(),
                grain.Batch(batch_size, drop_remainder=True),
            ]
        else:
            operations = [
                ParseMsgpackOp(label_offset=label_offset),
                DecodeOp(),
                CenterCropOp(size=image_size),
                NormalizeOp(),
                grain.Batch(batch_size, drop_remainder=True),
            ]

    log.info(f"DataLoader: shuffle={shuffle}, prefetch_workers={prefetch_workers}")

    return grain.DataLoader(
        data_source=source,
        sampler=grain.IndexSampler(
            num_records=len(source),
            num_epochs=None,  # Infinite iteration
            seed=seed,
            shuffle=shuffle,
            shard_options=grain.ShardOptions(
                worker_id, num_workers, drop_remainder=True
            ),
        ),
        operations=operations,
        worker_count=prefetch_workers,
    )
