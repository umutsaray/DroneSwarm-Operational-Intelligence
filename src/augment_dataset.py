from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.dataset import DatasetPair, discover_dataset
from src.utils import ensure_dir, load_config, read_yolo_label, resolve_path


@dataclass(frozen=True)
class PixelBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def yolo_to_pixel_boxes(boxes: list[tuple[int, float, float, float, float]], width: int, height: int) -> list[PixelBox]:
    pixel_boxes: list[PixelBox] = []
    for class_id, xc, yc, box_w, box_h in boxes:
        half_w = box_w * width / 2.0
        half_h = box_h * height / 2.0
        center_x = xc * width
        center_y = yc * height
        pixel_boxes.append(
            PixelBox(
                class_id=class_id,
                x1=center_x - half_w,
                y1=center_y - half_h,
                x2=center_x + half_w,
                y2=center_y + half_h,
            )
        )
    return pixel_boxes


def pixel_to_yolo_boxes(boxes: list[PixelBox], width: int, height: int, min_box_area: float) -> list[tuple[int, float, float, float, float]]:
    yolo_boxes: list[tuple[int, float, float, float, float]] = []
    for box in boxes:
        x1 = max(0.0, min(float(width), box.x1))
        y1 = max(0.0, min(float(height), box.y1))
        x2 = max(0.0, min(float(width), box.x2))
        y2 = max(0.0, min(float(height), box.y2))
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 1.0 or box_h <= 1.0:
            continue
        norm_w = box_w / width
        norm_h = box_h / height
        if norm_w * norm_h < min_box_area:
            continue
        xc = (x1 + x2) / 2.0 / width
        yc = (y1 + y2) / 2.0 / height
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < norm_w <= 1.0 and 0.0 < norm_h <= 1.0):
            continue
        yolo_boxes.append((box.class_id, xc, yc, norm_w, norm_h))
    return yolo_boxes


def horizontal_flip(image: Image.Image, boxes: list[PixelBox]) -> tuple[Image.Image, list[PixelBox]]:
    width, _ = image.size
    flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    transformed = [
        PixelBox(box.class_id, width - box.x2, box.y1, width - box.x1, box.y2)
        for box in boxes
    ]
    return flipped, transformed


def brightness_contrast(image: Image.Image, boxes: list[PixelBox], rng: random.Random) -> tuple[Image.Image, list[PixelBox]]:
    brightness = rng.uniform(0.82, 1.18)
    contrast = rng.uniform(0.85, 1.20)
    adjusted = ImageEnhance.Brightness(image).enhance(brightness)
    adjusted = ImageEnhance.Contrast(adjusted).enhance(contrast)
    return adjusted, boxes[:]


def gaussian_blur(image: Image.Image, boxes: list[PixelBox], rng: random.Random) -> tuple[Image.Image, list[PixelBox]]:
    radius = rng.uniform(0.4, 1.1)
    return image.filter(ImageFilter.GaussianBlur(radius=radius)), boxes[:]


def mild_noise(image: Image.Image, boxes: list[PixelBox], rng: random.Random) -> tuple[Image.Image, list[PixelBox]]:
    array = np.asarray(image.convert("RGB")).astype(np.int16)
    sigma = rng.uniform(3.0, 9.0)
    noise_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    noise = noise_rng.normal(0.0, sigma, size=array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy), boxes[:]


def rotate_box(box: PixelBox, width: int, height: int, angle_degrees: float) -> PixelBox:
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx = width / 2.0
    cy = height / 2.0
    corners = [(box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2)]
    rotated: list[tuple[float, float]] = []
    for x, y in corners:
        dx = x - cx
        dy = y - cy
        rx = cx + cos_a * dx - sin_a * dy
        ry = cy + sin_a * dx + cos_a * dy
        rotated.append((rx, ry))
    xs = [point[0] for point in rotated]
    ys = [point[1] for point in rotated]
    return PixelBox(box.class_id, min(xs), min(ys), max(xs), max(ys))


def slight_rotation(image: Image.Image, boxes: list[PixelBox], rng: random.Random) -> tuple[Image.Image, list[PixelBox]]:
    angle = rng.uniform(-8.0, 8.0)
    rotated_image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    width, height = image.size
    return rotated_image, [rotate_box(box, width, height, -angle) for box in boxes]


def scale_zoom(image: Image.Image, boxes: list[PixelBox], rng: random.Random) -> tuple[Image.Image, list[PixelBox]]:
    width, height = image.size
    scale = rng.uniform(1.05, 1.20)
    scaled_w = int(width * scale)
    scaled_h = int(height * scale)
    scaled = image.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
    left = (scaled_w - width) // 2
    top = (scaled_h - height) // 2
    cropped = scaled.crop((left, top, left + width, top + height))
    transformed = [
        PixelBox(
            box.class_id,
            box.x1 * scale - left,
            box.y1 * scale - top,
            box.x2 * scale - left,
            box.y2 * scale - top,
        )
        for box in boxes
    ]
    return cropped, transformed


def random_crop(image: Image.Image, boxes: list[PixelBox], rng: random.Random) -> tuple[Image.Image, list[PixelBox]]:
    width, height = image.size
    crop_scale = rng.uniform(0.88, 0.96)
    crop_w = int(width * crop_scale)
    crop_h = int(height * crop_scale)
    left = rng.randint(0, max(0, width - crop_w))
    top = rng.randint(0, max(0, height - crop_h))
    cropped = image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.BILINEAR)
    sx = width / crop_w
    sy = height / crop_h
    transformed = [
        PixelBox(
            box.class_id,
            (box.x1 - left) * sx,
            (box.y1 - top) * sy,
            (box.x2 - left) * sx,
            (box.y2 - top) * sy,
        )
        for box in boxes
    ]
    return cropped, transformed


AUGMENTATIONS = {
    "horizontal_flip": horizontal_flip,
    "brightness_contrast": brightness_contrast,
    "slight_rotation": slight_rotation,
    "scale_zoom": scale_zoom,
    "gaussian_blur": gaussian_blur,
    "mild_noise": mild_noise,
    "random_crop": random_crop,
}


def format_yolo_label(boxes: list[tuple[int, float, float, float, float]]) -> str:
    return "\n".join(
        f"{class_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"
        for class_id, xc, yc, width, height in boxes
    ) + ("\n" if boxes else "")


def copy_original_dataset(config: dict[str, Any], output_dir: Path) -> None:
    images_out = ensure_dir(output_dir / "images")
    labels_out = ensure_dir(output_dir / "labels")
    for pair in discover_dataset(config):
        image_target = images_out / pair.image.name
        label_target = labels_out / pair.label.name
        if not image_target.exists():
            image_target.write_bytes(pair.image.read_bytes())
        if not label_target.exists():
            label_target.write_bytes(pair.label.read_bytes())


def clear_previous_augmentations(output_dir: Path) -> None:
    for directory, pattern in ((output_dir / "images", "*_aug_*"), (output_dir / "labels", "*_aug_*")):
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def augment_pair(
    pair: DatasetPair,
    output_dir: Path,
    augmentation_type: str,
    copy_index: int,
    min_box_area: float,
    rng: random.Random,
) -> dict[str, Any]:
    images_out = ensure_dir(output_dir / "images")
    labels_out = ensure_dir(output_dir / "labels")
    with Image.open(pair.image) as raw_image:
        image = raw_image.convert("RGB")
    width, height = image.size
    original_boxes = read_yolo_label(pair.label)
    pixel_boxes = yolo_to_pixel_boxes(original_boxes, width, height)

    transform = AUGMENTATIONS[augmentation_type]
    if augmentation_type == "horizontal_flip":
        aug_image, aug_pixel_boxes = transform(image, pixel_boxes)  # type: ignore[misc]
    else:
        aug_image, aug_pixel_boxes = transform(image, pixel_boxes, rng)  # type: ignore[misc]
    aug_boxes = pixel_to_yolo_boxes(aug_pixel_boxes, width, height, min_box_area)

    stem = f"{pair.image.stem}_aug_{copy_index:02d}_{augmentation_type}"
    image_path = images_out / f"{stem}.png"
    label_path = labels_out / f"{stem}.txt"
    valid = bool(aug_boxes) or not original_boxes
    if not valid:
        return {
            "original_image": pair.image.name,
            "augmented_image": image_path.name,
            "augmentation_type": augmentation_type,
            "original_box_count": len(original_boxes),
            "augmented_box_count": 0,
            "valid": False,
        }

    aug_image.save(image_path)
    label_path.write_text(format_yolo_label(aug_boxes), encoding="utf-8")

    return {
        "original_image": pair.image.name,
        "augmented_image": image_path.name,
        "augmentation_type": augmentation_type,
        "original_box_count": len(original_boxes),
        "augmented_box_count": len(aug_boxes),
        "valid": valid,
    }


def augment_dataset(config: dict[str, Any]) -> dict[str, Any]:
    aug_cfg = config.get("augmentation", {})
    if not aug_cfg.get("enabled", True):
        raise RuntimeError("Augmentation is disabled in config. Set augmentation.enabled=true to run this command.")

    output_dir = Path(aug_cfg.get("output_dir", "data_augmented"))
    output_dir = resolve_path(output_dir)
    ensure_dir(output_dir / "images")
    ensure_dir(output_dir / "labels")
    ensure_dir(output_dir / "splits")
    clear_previous_augmentations(output_dir)

    pairs = discover_dataset(config)
    copy_original_dataset(config, output_dir)

    rng = random.Random(int(aug_cfg.get("seed", 42)))
    copies_per_image = int(aug_cfg.get("copies_per_image", 4))
    min_box_area = float(aug_cfg.get("min_box_area", 0.0001))
    augmentation_names = list(AUGMENTATIONS)
    rows: list[dict[str, Any]] = []

    for pair in pairs:
        for copy_index in range(copies_per_image):
            augmentation_type = augmentation_names[(copy_index + rng.randint(0, len(augmentation_names) - 1)) % len(augmentation_names)]
            rows.append(augment_pair(pair, output_dir, augmentation_type, copy_index + 1, min_box_area, rng))

    summary_path = output_dir / "augmentation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "original_image",
            "augmented_image",
            "augmentation_type",
            "original_box_count",
            "augmented_box_count",
            "valid",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["valid"]]
    return {
        "original_images": len(pairs),
        "augmented_images": len(rows),
        "valid_augmented_images": len(valid_rows),
        "output_dir": str(output_dir),
        "summary": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate YOLO-safe training augmentations.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    stats = augment_dataset(config)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
