import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.augment_dataset import (
    PixelBox,
    horizontal_flip,
    pixel_to_yolo_boxes,
    yolo_to_pixel_boxes,
)
from src.dataset import prepare_dataset
from src.utils import read_yolo_label


class AugmentationTests(unittest.TestCase):
    def test_horizontal_flip_bbox_correctness(self):
        image = Image.new("RGB", (100, 100), "white")
        boxes = [PixelBox(0, 10, 20, 30, 40)]
        _, flipped = horizontal_flip(image, boxes)
        self.assertEqual(flipped[0], PixelBox(0, 70, 20, 90, 40))

    def test_augmented_labels_remain_normalized(self):
        boxes = [(0, 0.25, 0.5, 0.2, 0.2)]
        pixels = yolo_to_pixel_boxes(boxes, 100, 100)
        _, flipped = horizontal_flip(Image.new("RGB", (100, 100)), pixels)
        normalized = pixel_to_yolo_boxes(flipped, 100, 100, min_box_area=0.0001)
        for _, xc, yc, width, height in normalized:
            self.assertGreaterEqual(xc, 0.0)
            self.assertLessEqual(xc, 1.0)
            self.assertGreaterEqual(yc, 0.0)
            self.assertLessEqual(yc, 1.0)
            self.assertGreater(width, 0.0)
            self.assertLessEqual(width, 1.0)
            self.assertGreater(height, 0.0)
            self.assertLessEqual(height, 1.0)

    def test_no_empty_label_file_when_valid_boxes_exist(self):
        boxes = [PixelBox(0, 10, 10, 30, 30)]
        normalized = pixel_to_yolo_boxes(boxes, 100, 100, min_box_area=0.0001)
        self.assertTrue(normalized)

    def test_val_test_are_not_augmented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "data" / "images"
            labels = root / "data" / "labels"
            aug_images = root / "data_augmented" / "images"
            aug_labels = root / "data_augmented" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            aug_images.mkdir(parents=True)
            aug_labels.mkdir(parents=True)
            for index in range(6):
                image_name = f"img_{index}.png"
                Image.new("RGB", (100, 100), "white").save(images / image_name)
                (labels / f"img_{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
                Image.new("RGB", (100, 100), "white").save(aug_images / f"img_{index}_aug_01_horizontal_flip.png")
                (aug_labels / f"img_{index}_aug_01_horizontal_flip.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            config = {
                "project": {"random_seed": 1},
                "paths": {
                    "data_dir": str(root / "data"),
                    "images_dir": str(images),
                    "labels_dir": str(labels),
                    "splits_dir": str(root / "data" / "splits"),
                },
                "dataset": {
                    "image_extensions": [".png"],
                    "train_ratio": 0.5,
                    "val_ratio": 0.25,
                    "test_ratio": 0.25,
                },
                "augmentation": {"output_dir": str(root / "data_augmented")},
            }
            prepare_dataset(config, use_augmented=True)
            val_lines = (root / "data_augmented" / "splits" / "val.txt").read_text(encoding="utf-8").splitlines()
            test_lines = (root / "data_augmented" / "splits" / "test.txt").read_text(encoding="utf-8").splitlines()
            self.assertTrue(val_lines)
            self.assertTrue(test_lines)
            self.assertFalse(any("_aug_" in line for line in val_lines))
            self.assertFalse(any("_aug_" in line for line in test_lines))


if __name__ == "__main__":
    unittest.main()
