import tempfile
import unittest
from pathlib import Path

from src.utils import read_yolo_label


class DatasetUtilityTests(unittest.TestCase):
    def test_yolo_label_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "sample.txt"
            label_path.write_text("0 0.5 0.4 0.2 0.1\n", encoding="utf-8")
            boxes = read_yolo_label(label_path, strict=True)
        self.assertEqual(boxes, [(0, 0.5, 0.4, 0.2, 0.1)])


if __name__ == "__main__":
    unittest.main()
