from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from src.dataset import discover_dataset
from src.utils import config_path, load_config, model_configs, resolve_path


def _ok(message: str) -> tuple[bool, str]:
    return True, f"[OK] {message}"


def _fail(message: str) -> tuple[bool, str]:
    return False, f"[FAIL] {message}"


def _warn(message: str) -> tuple[bool, str]:
    return True, f"[WARN] {message}"


def check_splits(config: dict[str, Any]) -> list[tuple[bool, str]]:
    splits_dir = config_path(config, "splits_dir")
    results: list[tuple[bool, str]] = []
    for name in ("train.txt", "val.txt", "test.txt"):
        path = splits_dir / name
        if not path.exists():
            results.append(_fail(f"Missing split file: {path}"))
            continue
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            results.append(_fail(f"Split file is empty: {path}"))
            continue
        non_absolute = [line for line in lines if not Path(line).is_absolute()]
        missing_images = [line for line in lines if not Path(line).exists()]
        if non_absolute:
            results.append(_fail(f"Split file contains non-absolute paths: {path}"))
        elif missing_images:
            results.append(_fail(f"Split file references missing images: {path}"))
        else:
            results.append(_ok(f"{name} exists with {len(lines)} absolute image paths"))
    data_yaml = splits_dir / "data.yaml"
    if not data_yaml.exists():
        results.append(_fail(f"Missing YOLO data.yaml: {data_yaml}"))
    else:
        text = data_yaml.read_text(encoding="utf-8")
        forbidden = ["path: ..", "train: splits/train.txt", "val: splits/val.txt", "test: splits/test.txt"]
        if any(item in text for item in forbidden):
            results.append(_fail("data/splits/data.yaml still contains relative path entries"))
        elif "\\" in text:
            results.append(_fail("data/splits/data.yaml contains backslash paths; use absolute POSIX paths"))
        else:
            entries: dict[str, str] = {}
            for raw_line in text.splitlines():
                if ":" not in raw_line:
                    continue
                key, value = raw_line.split(":", maxsplit=1)
                if key.strip() in {"train", "val", "test"}:
                    entries[key.strip()] = value.strip()
            invalid = [f"{key}: {value}" for key, value in entries.items() if not Path(value).is_absolute()]
            missing = [f"{key}: {value}" for key, value in entries.items() if not Path(value).exists()]
            if set(entries) != {"train", "val", "test"}:
                results.append(_fail("data/splits/data.yaml must contain train, val, and test entries"))
            elif invalid:
                results.append(_fail("data/splits/data.yaml contains non-absolute entries: " + "; ".join(invalid)))
            elif missing:
                results.append(_fail("data/splits/data.yaml references missing split files: " + "; ".join(missing)))
            else:
                results.append(_ok(f"YOLO data.yaml exists with absolute POSIX split references: {data_yaml}"))
    return results


def check_model_readiness(config: dict[str, Any]) -> tuple[bool, str]:
    ready_messages: list[str] = []
    missing_messages: list[str] = []
    models_dir = config_path(config, "models_dir")
    for model in model_configs(config):
        name = str(model["name"])
        trained = models_dir / name / "best.pt"
        base = resolve_path(model["base_weights"])
        base_name = str(model["base_weights"]).replace("\\", "/")
        if trained.exists():
            ready_messages.append(f"{name}: trained weights found")
        elif base.exists():
            ready_messages.append(f"{name}: base weights found, training can start")
        elif base_name in {"yolov8n.pt", "yolov8s.pt"}:
            ready_messages.append(f"{name}: pretrained {base_name} is not bundled; Ultralytics can download it during training")
        else:
            missing_messages.append(f"{name}: missing {trained} and {base}")
    if missing_messages:
        return _fail("; ".join(missing_messages))
    return _ok("; ".join(ready_messages))


def run_checks(config: dict[str, Any]) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    images_dir = config_path(config, "images_dir")
    labels_dir = config_path(config, "labels_dir")
    paper_outputs = config_path(config, "paper_outputs_dir")
    requirements = resolve_path("requirements.txt")

    results.append(_ok(f"data/images exists: {images_dir}") if images_dir.exists() else _fail(f"Missing data/images: {images_dir}"))
    results.append(_ok(f"data/labels exists: {labels_dir}") if labels_dir.exists() else _fail(f"Missing data/labels: {labels_dir}"))

    try:
        pairs = discover_dataset(config)
        results.append(_ok(f"Found {len(pairs)} matching image-label pairs"))
    except Exception as exc:
        results.append(_fail(f"Image-label pair validation failed: {exc}"))

    results.extend(check_splits(config))
    results.append(*[check_model_readiness(config)])
    results.append(_ok(f"paper_outputs exists: {paper_outputs}") if paper_outputs.exists() else _fail(f"Missing paper_outputs: {paper_outputs}"))
    results.append(_ok(f"requirements.txt exists: {requirements}") if requirements.exists() else _fail(f"Missing requirements.txt: {requirements}"))

    if importlib.util.find_spec("ultralytics") is None:
        results.append(_warn("Ultralytics is not installed in this environment; install requirements before training."))
    else:
        results.append(_ok("Ultralytics import is available"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GitHub-ready project reproducibility.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_checks(config)
    failed = False
    for ok, message in results:
        print(message)
        failed = failed or not ok
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
