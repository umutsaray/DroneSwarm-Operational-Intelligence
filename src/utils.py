from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load project config.

    The checked-in config is JSON-compatible YAML so the project can bootstrap
    even before PyYAML is installed. If users switch to normal YAML syntax,
    PyYAML from requirements.txt is used.
    """
    config_path = resolve_path(path or DEFAULT_CONFIG_PATH)
    text = config_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "config.yaml uses YAML syntax but PyYAML is not installed. "
                "Install requirements.txt or keep config.yaml JSON-compatible."
            ) from exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping: {config_path}")
        return data


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def config_path(config: dict[str, Any], key: str) -> Path:
    try:
        value = config["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Missing paths.{key} in config") from exc
    return resolve_path(value)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(config: dict[str, Any], run_name: str | None = None) -> Path:
    run_id = run_name or f"run_{timestamp()}"
    run_dir = ensure_dir(config_path(config, "experiments_dir") / run_id)
    for child in ("metrics", "predictions", "figures", "weights", "reports"):
        ensure_dir(run_dir / child)
    return run_dir


def latest_run_dir(config: dict[str, Any]) -> Path | None:
    experiments_dir = config_path(config, "experiments_dir")
    runs = sorted(experiments_dir.glob("run_*"), key=lambda p: p.name)
    return runs[-1] if runs else None


def model_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = config.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError("config.models must contain at least one model entry")
    return models


def get_model_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    for model in model_configs(config):
        if model.get("name") == model_name:
            return model
    raise KeyError(f"Unknown model in config: {model_name}")


def list_image_files(directory: str | Path, extensions: Iterable[str]) -> list[Path]:
    directory = Path(directory)
    suffixes = {ext.lower() for ext in extensions}
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )


def read_yolo_label(label_path: str | Path, strict: bool = False) -> list[tuple[int, float, float, float, float]]:
    """Read normalized YOLO labels as (class_id, x_center, y_center, width, height)."""
    path = Path(label_path)
    boxes: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        if strict:
            raise FileNotFoundError(path)
        return boxes

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            if strict:
                raise ValueError(f"{path}:{line_number} has fewer than 5 YOLO fields")
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, width, height = (float(v) for v in parts[1:5])
        except ValueError:
            if strict:
                raise
            continue
        boxes.append((cls, xc, yc, width, height))
    return boxes


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def relative_posix(path: str | Path, start: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(start).resolve()).as_posix()


def copy_if_exists(source: str | Path, destination: str | Path) -> bool:
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        return False
    ensure_dir(destination.parent)
    destination.write_bytes(source.read_bytes())
    return True
