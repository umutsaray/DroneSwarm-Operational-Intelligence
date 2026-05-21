from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from src.dataset import prepare_dataset
from src.utils import (
    config_path,
    copy_if_exists,
    create_run_dir,
    ensure_dir,
    get_model_config,
    load_config,
    model_configs,
    resolve_path,
    write_json,
)


FIGURE_MAP = {
    "confusion_matrix.png": "confusion_matrix.png",
    "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
    "results.png": "training_curves.png",
    "BoxPR_curve.png": "PR_curve.png",
    "BoxF1_curve.png": "F1_curve.png",
    "BoxP_curve.png": "P_curve.png",
    "BoxR_curve.png": "R_curve.png",
}


def selected_model_configs(config: dict[str, Any], names: list[str] | None) -> list[dict[str, Any]]:
    if not names:
        return model_configs(config)
    return [get_model_config(config, name) for name in names]


def _copy_training_artifacts(model_name: str, source_dir: Path, run_dir: Path, models_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    weights_dir = ensure_dir(run_dir / "weights")
    metrics_dir = ensure_dir(run_dir / "metrics")
    figures_dir = ensure_dir(run_dir / "figures")
    model_dir = ensure_dir(models_dir / model_name)

    for weight_name in ("best.pt", "last.pt"):
        source = source_dir / "weights" / weight_name
        run_target = weights_dir / f"{model_name}_{weight_name}"
        model_target = model_dir / weight_name
        if copy_if_exists(source, run_target):
            copied[f"run_{weight_name}"] = str(run_target)
            shutil.copy2(run_target, model_target)
            copied[f"model_{weight_name}"] = str(model_target)

    if copy_if_exists(source_dir / "results.csv", metrics_dir / f"results_{model_name}.csv"):
        copied["results_csv"] = str(metrics_dir / f"results_{model_name}.csv")
    if copy_if_exists(source_dir / "args.yaml", metrics_dir / f"training_config_{model_name}.yaml"):
        copied["training_config"] = str(metrics_dir / f"training_config_{model_name}.yaml")

    for source_name, target_suffix in FIGURE_MAP.items():
        target = figures_dir / f"{model_name}_{target_suffix}"
        if copy_if_exists(source_dir / source_name, target):
            copied[source_name] = str(target)
    return copied


def train_models(
    config: dict[str, Any],
    model_names: list[str] | None = None,
    run_name: str | None = None,
    prepare: bool = True,
    use_augmented: bool = False,
) -> Path:
    if prepare:
        prepare_dataset(config, overwrite=False, use_augmented=use_augmented)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install requirements.txt before training."
        ) from exc

    run_dir = create_run_dir(config, run_name)
    write_json(run_dir / "reports" / "run_config.json", config)
    if use_augmented:
        aug_dir = config["augmentation"].get("output_dir", "data_augmented")
        data_yaml = resolve_path(aug_dir) / "splits" / "data.yaml"
    else:
        data_yaml = config_path(config, "splits_dir") / "data.yaml"
    training_cfg = config["training"]
    models_dir = config_path(config, "models_dir")
    manifest: dict[str, Any] = {"run_dir": str(run_dir), "models": {}}

    for model_cfg in selected_model_configs(config, model_names):
        model_name = str(model_cfg["name"])
        base_weights = resolve_path(model_cfg["base_weights"])
        weights_arg = str(base_weights if base_weights.exists() else model_cfg["base_weights"])
        model = YOLO(weights_arg)

        result = model.train(
            data=str(data_yaml),
            epochs=int(training_cfg["epochs"]),
            imgsz=int(training_cfg["imgsz"]),
            batch=int(training_cfg["batch"]),
            device=training_cfg.get("device", "cpu"),
            workers=int(training_cfg.get("workers", 0)),
            patience=int(training_cfg.get("patience", 50)),
            project=str(run_dir / "ultralytics"),
            name=model_name,
            exist_ok=True,
            pretrained=bool(training_cfg.get("pretrained", True)),
        )

        source_dir = Path(getattr(result, "save_dir", run_dir / "ultralytics" / model_name))
        artifacts = _copy_training_artifacts(model_name, source_dir, run_dir, models_dir)
        summary_path = run_dir / "reports" / f"model_summary_{model_name}.txt"
        try:
            summary = model.info(detailed=True, verbose=False)
        except Exception as exc:  # Ultralytics versions differ here.
            summary = f"Model summary unavailable: {exc}"
        summary_path.write_text(str(summary), encoding="utf-8")
        artifacts["model_summary"] = str(summary_path)
        manifest["models"][model_name] = artifacts

    write_json(run_dir / "reports" / "run_manifest.json", manifest)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n/YOLOv8s with reproducible splits.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--models", nargs="*", help="Optional model names, e.g. yolov8n yolov8s")
    parser.add_argument("--run-name", help="Optional run directory name under experiments/.")
    parser.add_argument("--skip-prepare", action="store_true", help="Use existing data/splits without preparing dataset.")
    parser.add_argument("--use-augmented", action="store_true", help="Train with data_augmented splits: original train plus augmented train, original val/test.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = train_models(
        config,
        model_names=args.models,
        run_name=args.run_name,
        prepare=not args.skip_prepare,
        use_augmented=args.use_augmented,
    )
    print(f"Training artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
