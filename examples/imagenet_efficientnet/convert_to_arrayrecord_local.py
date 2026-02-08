import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import msgpack


def parse_tfrecord_example(raw_record: bytes) -> dict:
    """Parse a TFRecord example into image bytes and label.

    Uses TensorFlow only for parsing the protobuf format.
    """
    import tensorflow as tf

    features = tf.io.parse_single_example(
        raw_record,
        features={
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/class/label": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    return {
        "image": features["image/encoded"].numpy(),
        "label": int(features["image/class/label"].numpy()),
    }


def convert_single_file(args: tuple) -> tuple[str, int, bool]:
    """Convert a single TFRecord file to ArrayRecord.

    Args:
        args: Tuple of (tfrecord_path, output_path)

    Returns:
        Tuple of (input_path, record_count, was_skipped)
    """
    import tensorflow as tf
    from array_record.python.array_record_module import ArrayRecordWriter

    tfrecord_path, output_path = args

    # Skip if already exists
    if os.path.exists(output_path):
        try:
            from array_record.python.array_record_module import ArrayRecordReader
            reader = ArrayRecordReader(output_path)
            count = reader.num_records()
            reader.close()
            return tfrecord_path, count, True
        except Exception:
            # File exists but is corrupted, re-convert
            os.remove(output_path)

    # Convert
    writer = ArrayRecordWriter(output_path, options="group_size:1")
    count = 0

    for raw_record in tf.data.TFRecordDataset(tfrecord_path):
        example = parse_tfrecord_example(raw_record.numpy())
        packed = msgpack.packb(example, use_bin_type=True)
        writer.write(packed)
        count += 1

    writer.close()
    return tfrecord_path, count, False


def main():
    parser = argparse.ArgumentParser(
        description="Convert TFRecords to ArrayRecord format (local version)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Glob pattern for TFRecords (e.g., /path/to/train/*.tfrecord)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory (e.g., /path/to/train_arrayrecord/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    args = parser.parse_args()

    # Find all input files
    files = sorted(glob.glob(args.input))
    if not files:
        print(f"No files found matching {args.input}")
        return 1

    print(f"Found {len(files)} TFRecord files")
    print(f"Output directory: {args.output}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Prepare conversion tasks
    tasks = []
    for tfrecord_path in files:
        basename = os.path.basename(tfrecord_path)
        if ".tfrecord" in basename:
            basename = basename.replace(".tfrecord", ".arrayrecord")
        else:
            basename = basename + ".arrayrecord"
        output_path = os.path.join(args.output, basename)
        tasks.append((tfrecord_path, output_path))

    # Convert in parallel using ProcessPoolExecutor
    # (ProcessPoolExecutor avoids TensorFlow threading issues)
    total_records = 0
    completed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_single_file, task): task[0] for task in tasks}

        for future in as_completed(futures):
            try:
                path, count, was_skipped = future.result()
                total_records += count
                completed += 1
                if was_skipped:
                    skipped += 1
                    status = "skipped (exists)"
                else:
                    status = f"{count} records"
                print(f"[{completed}/{len(files)}] {os.path.basename(path)}: {status}")
            except Exception as e:
                print(f"ERROR converting {futures[future]}: {e}")
                raise

    print()
    print("Conversion complete!")
    print(f"  Files: {len(files)} ({skipped} skipped)")
    print(f"  Total records: {total_records:,}")
    print(f"  Output: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())


"""
python examples/imagenet_efficientnet/convert_to_arrayrecord_local.py \
    --input "archive/train/*.tfrecord" \
    --output "archive/train_arrayrecord/" \
    --workers 16
    
python examples/imagenet_efficientnet/convert_to_arrayrecord_local.py \
    --input "archive/validation/*.tfrecord" \
    --output "archive/validation_arrayrecord/" \
    --workers 16       
"""