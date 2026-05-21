from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import (
    config_path,
    ensure_dir,
    list_image_files,
    load_config,
    read_yolo_label,
    resolve_path,
)


@dataclass(frozen=True)
class DatasetPair:
    image: Path
    label: Path
    image_id: str
    drone_count: int


def validate_dataset_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    images_dir = config_path(config, "images_dir")
    labels_dir = config_path(config, "labels_dir")
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Missing project-local image directory: {images_dir}")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing project-local label directory: {labels_dir}")
    return images_dir, labels_dir


def discover_dataset(config: dict[str, Any]) -> list[DatasetPair]:
    images_dir, labels_dir = validate_dataset_dirs(config)
    extensions = config["dataset"]["image_extensions"]
    image_files = list_image_files(images_dir, extensions)
    label_files = sorted(labels_dir.glob("*.txt"))

    image_stems = {image.stem for image in image_files}
    label_stems = {label.stem for label in label_files}
    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    if missing_labels or missing_images:
        details = []
        if missing_labels:
            details.append(f"images without labels: {', '.join(missing_labels[:10])}")
        if missing_images:
            details.append(f"labels without images: {', '.join(missing_images[:10])}")
        raise ValueError("Image-label pairs do not match: " + "; ".join(details))

    pairs: list[DatasetPair] = []
    for image in image_files:
        label = labels_dir / f"{image.stem}.txt"
        boxes = read_yolo_label(label)
        pairs.append(DatasetPair(image=image.resolve(), label=label.resolve(), image_id=image.stem, drone_count=len(boxes)))

    if not pairs:
        raise ValueError(f"No image-label pairs found in {images_dir} and {labels_dir}")
    return pairs


def split_dataset(
    pairs: list[DatasetPair],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[DatasetPair]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    if n_total >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n_total:
            n_train = n_total - 2
            n_val = 1

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    if n_total >= 3 and not test:
        test = [val.pop()]

    return {"train": train, "val": val, "test": test}


def absolute_posix(path: Path) -> str:
    return path.resolve().as_posix()


def write_split_files(
    config: dict[str, Any],
    splits: dict[str, list[DatasetPair]],
    splits_dir: Path | None = None,
) -> Path:
    splits_dir = ensure_dir(splits_dir or config_path(config, "splits_dir"))
    split_paths: dict[str, Path] = {}

    for split_name, rows in splits.items():
        split_file = splits_dir / f"{split_name}.txt"
        split_paths[split_name] = split_file.resolve()
        lines = [absolute_posix(pair.image) for pair in rows]
        split_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_yaml = splits_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"train: {absolute_posix(split_paths['train'])}",
                f"val: {absolute_posix(split_paths['val'])}",
                f"test: {absolute_posix(split_paths['test'])}",
                "nc: 1",
                "names:",
                "  0: drone",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary_path = splits_dir / "split_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image_id", "image", "label", "drone_count"])
        writer.writeheader()
        for split_name, rows in splits.items():
            for pair in rows:
                writer.writerow(
                    {
                        "split": split_name,
                        "image_id": pair.image_id,
                        "image": absolute_posix(pair.image),
                        "label": absolute_posix(pair.label),
                        "drone_count": pair.drone_count,
                    }
                )
    return data_yaml


def augmentation_output_dir(config: dict[str, Any]) -> Path:
    output_dir = Path(config.get("augmentation", {}).get("output_dir", "data_augmented"))
    return resolve_path(output_dir)


def augmented_pairs_for_train(train_pairs: list[DatasetPair], config: dict[str, Any]) -> list[DatasetPair]:
    output_dir = augmentation_output_dir(config)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Augmented dataset not found at {output_dir}. Run python -m src.augment_dataset --config config/config.yaml first."
        )

    augmented: list[DatasetPair] = []
    for pair in train_pairs:
        pattern = f"{pair.image.stem}_aug_*"
        for image in sorted(images_dir.glob(pattern + ".png")):
            label = labels_dir / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing augmented label for {image}: {label}")
            boxes = read_yolo_label(label)
            augmented.append(DatasetPair(image=image.resolve(), label=label.resolve(), image_id=image.stem, drone_count=len(boxes)))
    return augmented


def prepare_dataset(config: dict[str, Any], overwrite: bool = False, use_augmented: bool = False) -> dict[str, Any]:
    # The project is self-contained: data/images and data/labels must already
    # exist under the project root. overwrite is accepted for CLI compatibility,
    # but no external copy step is performed.
    _ = overwrite
    pairs = discover_dataset(config)
    split_cfg = config["dataset"]
    splits = split_dataset(
        pairs,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(config["project"]["random_seed"]),
    )
    if use_augmented:
        augmented_train = augmented_pairs_for_train(splits["train"], config)
        output_dir = augmentation_output_dir(config)
        augmented_splits = {
            "train": [*splits["train"], *augmented_train],
            "val": splits["val"],
            "test": splits["test"],
        }
        data_yaml = write_split_files(config, augmented_splits, splits_dir=output_dir / "splits")
        return {
            "total_pairs": len(pairs),
            "original_train": len(splits["train"]),
            "original_val": len(splits["val"]),
            "original_test": len(splits["test"]),
            "augmented_train": len(augmented_train),
            "train": len(augmented_splits["train"]),
            "val": len(augmented_splits["val"]),
            "test": len(augmented_splits["test"]),
            "data_yaml": str(data_yaml),
            "academic_note": "The training set was expanded through augmentation, while validation and test sets were kept unchanged.",
        }

    data_yaml = write_split_files(config, splits)
    return {
        "total_pairs": len(pairs),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "data_yaml": str(data_yaml),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate project-local data and prepare YOLO split files.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for compatibility; no external copy is performed.")
    parser.add_argument("--use-augmented", action="store_true", help="Write data_augmented splits where train includes original train plus augmented train images.")
    args = parser.parse_args()

    config = load_config(args.config)
    stats = prepare_dataset(config, overwrite=args.overwrite, use_augmented=args.use_augmented)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
