from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from PIL import Image

from src.threat_score import calculate_threat_score
from src.utils import (
    config_path,
    ensure_dir,
    get_model_config,
    json_cell,
    latest_run_dir,
    load_config,
    model_configs,
    resolve_path,
    write_rows_csv,
)
from src.visualization import draw_boxes

PREDICTION_COLUMNS = [
    "image_filename",
    "image_path",
    "model_name",
    "number_of_detections",
    "average_confidence",
    "bounding_boxes",
    "predicted_classes",
    "confidences",
    "inference_time",
    "FPS",
    "N_norm",
    "A_swarm",
    "D",
    "P",
    "final_TS",
    "risk_level",
    "D_raw",
    "d",
    "centroid_x",
    "centroid_y",
]


def test_image_paths(config: dict[str, Any]) -> list[Path]:
    split_file = config_path(config, "splits_dir") / "test.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing test split file: {split_file}. Run python -m src.dataset first.")
    data_dir = config_path(config, "data_dir")
    paths: list[Path] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            image_path = Path(line)
            paths.append(image_path.resolve() if image_path.is_absolute() else (data_dir / image_path).resolve())
    return paths


def model_weight_path(config: dict[str, Any], model_name: str, explicit_path: str | Path | None = None) -> Path:
    if explicit_path:
        path = resolve_path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    models_dir = config_path(config, "models_dir")
    preferred = models_dir / model_name / "best.pt"
    if preferred.exists():
        return preferred

    run_dir = latest_run_dir(config)
    if run_dir:
        candidate = run_dir / "weights" / f"{model_name}_best.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No trained weight found for {model_name}. Expected {preferred} or latest run weights."
    )


def _boxes_from_result(result: Any, class_ids: list[int]) -> list[dict[str, float]]:
    if result.boxes is None:
        return []
    xywhn = result.boxes.xywhn.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    boxes: list[dict[str, float]] = []
    allowed = set(int(c) for c in class_ids)
    for raw_class, raw_box, raw_confidence in zip(classes, xywhn, confidences):
        class_id = int(raw_class)
        if allowed and class_id not in allowed:
            continue
        xc, yc, width, height = (float(v) for v in raw_box)
        boxes.append(
            {
                "class_id": class_id,
                "x_center": xc,
                "y_center": yc,
                "width": width,
                "height": height,
                "confidence": float(raw_confidence),
            }
        )
    return boxes


def predict_image(
    model: Any,
    image: Image.Image,
    model_name: str,
    class_ids: list[int],
    score_config: dict[str, Any],
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[list[dict[str, float]], dict[str, Any], float]:
    start = time.perf_counter()
    results = model.predict(image.convert("RGB"), conf=confidence_threshold, iou=iou_threshold, verbose=False)
    elapsed = time.perf_counter() - start
    result = results[0]
    boxes = _boxes_from_result(result, class_ids)
    metrics = calculate_threat_score(boxes, score_config)
    metrics["model_name"] = model_name
    return boxes, metrics, elapsed


def prediction_row(
    image_path: Path,
    model_name: str,
    boxes: list[dict[str, float]],
    metrics: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    confidences = [round(float(box["confidence"]), 6) for box in boxes]
    classes = [int(box["class_id"]) for box in boxes]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "image_filename": image_path.name,
        "image_path": str(image_path),
        "model_name": model_name,
        "number_of_detections": len(boxes),
        "average_confidence": round(avg_confidence, 6),
        "bounding_boxes": json_cell(boxes),
        "predicted_classes": json_cell(classes),
        "confidences": json_cell(confidences),
        "inference_time": round(elapsed, 6),
        "FPS": round(1.0 / elapsed, 6) if elapsed > 0 else 0.0,
        "N_norm": metrics["N_norm"],
        "A_swarm": metrics["A_swarm"],
        "D": metrics["D"],
        "P": metrics["P"],
        "final_TS": metrics["final_TS"],
        "risk_level": metrics["risk_level"],
        "D_raw": metrics["D_raw"],
        "d": "" if metrics["d"] is None else metrics["d"],
        "centroid_x": "" if metrics["centroid_x"] is None else metrics["centroid_x"],
        "centroid_y": "" if metrics["centroid_y"] is None else metrics["centroid_y"],
    }


def run_dataset_inference(
    config: dict[str, Any],
    model_names: list[str] | None = None,
    run_dir: str | Path | None = None,
    save_examples: int = 6,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install requirements.txt before inference."
        ) from exc

    if run_dir is None:
        resolved_run = latest_run_dir(config)
        if resolved_run is None:
            raise FileNotFoundError("No experiment run found. Train first or pass --run-dir.")
    else:
        resolved_run = resolve_path(run_dir)
    ensure_dir(resolved_run / "predictions")
    ensure_dir(resolved_run / "figures")

    selected = model_names or [model["name"] for model in model_configs(config)]
    image_paths = test_image_paths(config)
    inference_cfg = config["inference"]

    for model_name in selected:
        model_cfg = get_model_config(config, model_name)
        weights = model_weight_path(config, model_name)
        model = YOLO(str(weights))
        rows: list[dict[str, Any]] = []
        example_paths: list[Path] = []

        for index, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                boxes, metrics, elapsed = predict_image(
                    model=model,
                    image=image,
                    model_name=model_name,
                    class_ids=list(model_cfg.get("drone_class_ids", [0])),
                    score_config=config,
                    confidence_threshold=float(inference_cfg["confidence_threshold"]),
                    iou_threshold=float(inference_cfg["iou_threshold"]),
                )
                rows.append(prediction_row(image_path, model_name, boxes, metrics, elapsed))
                if index < save_examples:
                    annotated = draw_boxes(image, boxes)
                    example_path = resolved_run / "figures" / f"{model_name}_example_{index + 1:02d}_{image_path.name}"
                    annotated.save(example_path)
                    example_paths.append(example_path)

        write_rows_csv(
            resolved_run / "predictions" / f"predictions_{model_name}.csv",
            rows,
            PREDICTION_COLUMNS,
        )

    return resolved_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset-level YOLO inference and threat scoring.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--models", nargs="*", help="Optional model names, e.g. yolov8n yolov8s")
    parser.add_argument("--run-dir", help="Experiment run directory. Defaults to latest run.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = run_dataset_inference(config, model_names=args.models, run_dir=args.run_dir)
    print(f"Inference outputs saved to: {run_dir / 'predictions'}")


if __name__ == "__main__":
    main()
