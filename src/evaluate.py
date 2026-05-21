from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from src.utils import config_path, ensure_dir, latest_run_dir, load_config, model_configs, resolve_path, write_rows_csv
from src.visualization import (
    save_confusion_matrix_figure,
    save_dashboard_figure,
    save_dense_swarm_example,
    save_detection_comparison,
    save_system_architecture,
    save_threat_workflow,
    save_training_curves,
)

MODEL_COMPARISON_COLUMNS = [
    "Model",
    "Precision",
    "Recall",
    "mAP50",
    "mAP50_95",
    "Inference_Time_Mean",
    "FPS_Mean",
]

THREAT_SCORE_TABLE_COLUMNS = [
    "Image",
    "Model",
    "Detections",
    "Avg_Confidence",
    "A_swarm",
    "D",
    "P",
    "TS",
    "Risk_Level",
]

EXPERIMENT_CONFIGURATION_COLUMNS = ["Parameter", "Value"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{(key or "").strip(): value for key, value in row.items()} for row in reader]


def _last_results_row(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    return rows[-1] if rows else {}


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _format_optional(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def find_prediction_csv(run_dir: Path, model_name: str) -> Path | None:
    """Resolve prediction CSVs robustly inside an experiment run.

    The canonical location is run_dir/predictions/predictions_<model>.csv.
    Older or manually copied artifacts may be nested elsewhere, so fall back to
    a recursive search under the selected run directory.
    """
    filename = f"predictions_{model_name}.csv"
    preferred = run_dir / "predictions" / filename
    if preferred.exists():
        return preferred

    matches = sorted(
        (path for path in run_dir.rglob(filename) if path.is_file()),
        key=lambda path: (
            0 if path.parent.name == "predictions" else 1,
            len(path.parts),
            path.as_posix(),
        ),
    )
    return matches[0] if matches else None


def read_predictions(run_dir: Path, model_name: str) -> list[dict[str, str]]:
    prediction_csv = find_prediction_csv(run_dir, model_name)
    return _read_csv(prediction_csv) if prediction_csv else []


def read_training_history(run_dir: Path, model_name: str) -> list[dict[str, str]]:
    return _read_csv(run_dir / "metrics" / f"results_{model_name}.csv")


def collect_model_comparison(config: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_cfg in model_configs(config):
        model_name = str(model_cfg["name"])
        training = _last_results_row(run_dir / "metrics" / f"results_{model_name}.csv")
        predictions = read_predictions(run_dir, model_name)
        inference_time = _mean([_float(row, "inference_time") for row in predictions])
        fps = _mean([_float(row, "FPS") for row in predictions])
        rows.append(
            {
                "Model": model_name,
                "Precision": training.get("metrics/precision(B)", ""),
                "Recall": training.get("metrics/recall(B)", ""),
                "mAP50": training.get("metrics/mAP50(B)", ""),
                "mAP50_95": training.get("metrics/mAP50-95(B)", ""),
                "Inference_Time_Mean": _format_optional(inference_time),
                "FPS_Mean": _format_optional(fps),
            }
        )
    return rows


def collect_threat_scores(config: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_cfg in model_configs(config):
        model_name = str(model_cfg["name"])
        for row in read_predictions(run_dir, model_name):
            rows.append(
                {
                    "Image": row.get("image_filename", ""),
                    "Model": model_name,
                    "Detections": row.get("number_of_detections", ""),
                    "Avg_Confidence": row.get("average_confidence", ""),
                    "A_swarm": row.get("A_swarm", ""),
                    "D": row.get("D", ""),
                    "P": row.get("P", ""),
                    "TS": row.get("final_TS", ""),
                    "Risk_Level": row.get("risk_level", ""),
                }
            )
    return rows


def split_counts(config: dict[str, Any]) -> dict[str, int]:
    splits_dir = config_path(config, "splits_dir")
    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        path = splits_dir / f"{split_name}.txt"
        if not path.exists():
            counts[split_name] = 0
        else:
            counts[split_name] = len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    return counts


def augmented_split_counts(config: dict[str, Any]) -> dict[str, int]:
    output_dir = Path(config.get("augmentation", {}).get("output_dir", "data_augmented"))
    output_dir = resolve_path(output_dir)
    splits_dir = output_dir / "splits"
    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        path = splits_dir / f"{split_name}.txt"
        if not path.exists():
            counts[split_name] = 0
        else:
            counts[split_name] = len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    return counts


def _ram_gb() -> str:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return f"{pages * page_size / (1024 ** 3):.2f} GB"
    except (ValueError, OSError, AttributeError):
        pass
    return "Unknown (not measured)"


def _gpu_info() -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "Not detected by evaluation script", "Not detected by evaluation script"
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        return "Not detected by evaluation script", "Not detected by evaluation script"
    parts = [part.strip() for part in line.split(",", maxsplit=1)]
    if len(parts) == 1:
        return parts[0], "Unknown"
    return parts[0], parts[1]


def _framework_info() -> str:
    parts: list[str] = []
    try:
        import ultralytics  # type: ignore

        parts.append(f"Ultralytics {getattr(ultralytics, '__version__', 'installed')}")
    except ImportError:
        parts.append("Ultralytics not installed")
    try:
        import torch  # type: ignore

        parts.append(f"PyTorch {getattr(torch, '__version__', 'installed')}")
    except ImportError:
        parts.append("PyTorch not installed")
    return "; ".join(parts)


def collect_experiment_configuration(config: dict[str, Any]) -> list[dict[str, str]]:
    counts = split_counts(config)
    aug_counts = augmented_split_counts(config)
    augmented_train = max(0, aug_counts.get("train", 0) - counts["train"])
    gpu, gpu_memory = _gpu_info()
    rows = [
        ("Hardware Platform", platform.platform()),
        ("Processor", platform.processor() or "Unknown"),
        ("RAM", _ram_gb()),
        ("GPU", gpu),
        ("GPU Memory", gpu_memory),
        ("Framework", _framework_info()),
        ("Input Resolution", str(config["training"].get("imgsz", ""))),
        ("Batch Size", str(config["training"].get("batch", ""))),
        ("Models", ", ".join(str(model["name"]) for model in model_configs(config))),
        ("Epochs", str(config["training"].get("epochs", ""))),
        ("Train Images", str(counts["train"])),
        ("Validation Images", str(counts["val"])),
        ("Test Images", str(counts["test"])),
        ("Augmented Train Images", str(augmented_train)),
        ("Total Training Images After Augmentation", str(aug_counts.get("train", 0) or counts["train"])),
        ("Validation/Test Augmented", "No"),
    ]
    return [{"Parameter": key, "Value": value} for key, value in rows]


def find_confusion_matrix(run_dir: Path, model_name: str) -> Path | None:
    candidates = [
        run_dir / "figures" / f"{model_name}_confusion_matrix.png",
        run_dir / "ultralytics" / model_name / "confusion_matrix.png",
    ]
    candidates.extend(sorted((run_dir / "ultralytics").glob(f"**/{model_name}*/confusion_matrix.png")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_histories(config: dict[str, Any], run_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {str(model["name"]): read_training_history(run_dir, str(model["name"])) for model in model_configs(config)}


def highest_threat_row(config: dict[str, Any], run_dir: Path) -> dict[str, str] | None:
    all_rows: list[dict[str, str]] = []
    for model_cfg in model_configs(config):
        all_rows.extend(read_predictions(run_dir, str(model_cfg["name"])))
    if not all_rows:
        return None
    return max(all_rows, key=lambda row: _float(row, "final_TS") or -1.0)


def best_model_from_comparison(rows: list[dict[str, Any]]) -> str:
    scored: list[tuple[float, str]] = []
    for row in rows:
        for key in ("mAP50_95", "mAP50"):
            try:
                value = float(row.get(key, "") or "")
                scored.append((value, str(row.get("Model", ""))))
                break
            except ValueError:
                continue
    if not scored:
        return "Not determined (training metrics missing)"
    return max(scored)[1]


def build_experiment_summary(
    config: dict[str, Any],
    run_dir: Path | None,
    comparison: list[dict[str, Any]],
    threat_scores: list[dict[str, Any]],
    warnings: list[str],
    generated_files: list[Path],
) -> str:
    counts = split_counts(config)
    aug_counts = augmented_split_counts(config)
    augmented_train = max(0, aug_counts.get("train", 0) - counts["train"])
    dataset_size = counts["train"] + counts["val"] + counts["test"]
    trained_models = [row["Model"] for row in comparison if row.get("Precision") or row.get("mAP50")]
    best_model = best_model_from_comparison(comparison)
    highest = None
    if threat_scores:
        highest = max(threat_scores, key=lambda row: float(row.get("TS") or 0.0))

    lines = [
        "# Experiment Summary",
        "",
        f"Experiment run: `{run_dir}`" if run_dir else "Experiment run: not available",
        "",
        "## Dataset",
        "",
        f"- Dataset size: {dataset_size}",
        f"- Train images: {counts['train']}",
        f"- Validation images: {counts['val']}",
        f"- Test images: {counts['test']}",
        f"- Augmented train images: {augmented_train}",
        f"- Total training images after augmentation: {aug_counts.get('train', 0) or counts['train']}",
        "- Academic note: The training set was expanded through augmentation, while validation and test sets were kept unchanged.",
        "",
        "## Trained Models",
        "",
        f"- Trained models with available metrics: {', '.join(trained_models) if trained_models else 'None detected'}",
        f"- Best model: {best_model}",
        "",
        "## Performance Comparison",
        "",
    ]
    if comparison:
        for row in comparison:
            lines.append(
                f"- {row['Model']}: Precision={row.get('Precision', '') or 'missing'}, "
                f"Recall={row.get('Recall', '') or 'missing'}, "
                f"mAP50={row.get('mAP50', '') or 'missing'}, "
                f"mAP50-95={row.get('mAP50_95', '') or 'missing'}, "
                f"FPS mean={row.get('FPS_Mean', '') or 'missing'}"
            )
    else:
        lines.append("- No model comparison rows available.")

    lines.extend(["", "## Inference Summary", ""])
    if threat_scores:
        lines.append(f"- Prediction-derived threat rows: {len(threat_scores)}")
        lines.append(f"- Highest threat image: {highest['Image']} ({highest['Model']}), TS={highest['TS']}, risk={highest['Risk_Level']}")
    else:
        lines.append("- No prediction-derived threat rows available. Run dataset inference first.")

    lines.extend(["", "## Generated Figures and Tables", ""])
    for file_path in generated_files:
        lines.append(f"- `{file_path}`")

    lines.extend(["", "## Missing Data Warnings", ""])
    if warnings:
        for warning in sorted(set(warnings)):
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Remaining Limitations",
            "",
            "- Threat-score weights and thresholds still require domain validation.",
            "- Image-normalized distance and area are not physical-world measurements.",
            "- Missing artifacts are never replaced with simulated values.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs_report(
    root_report: Path,
    output_dir: Path,
    run_dir: Path | None,
    generated_files: list[Path],
    warnings: list[str],
) -> None:
    lines = [
        "# PAPER OUTPUTS REPORT",
        "",
        f"Output folder: `{output_dir}`",
        f"Experiment run: `{run_dir}`" if run_dir else "Experiment run: not available",
        "",
        "## Generated Figure Paths",
        "",
    ]
    for file_path in generated_files:
        if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            lines.append(f"- `{file_path}`")
    lines.extend(["", "## Generated Table Paths", ""])
    for file_path in generated_files:
        if file_path.suffix.lower() == ".csv":
            lines.append(f"- `{file_path}`")
    lines.extend(["", "## Missing Data Warnings", ""])
    if warnings:
        for warning in sorted(set(warnings)):
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Exact Commands",
            "",
            "```bash",
            "python -m src.train --config config/config.yaml",
            "python -m src.infer --config config/config.yaml --run-dir experiments/run_<timestamp>",
            "python -m src.evaluate --config config/config.yaml --run-dir experiments/run_<timestamp>",
            "```",
            "",
        ]
    )
    root_report.write_text("\n".join(lines), encoding="utf-8")


def generate_paper_outputs(config: dict[str, Any], run_dir: str | Path | None = None) -> Path:
    resolved_run = resolve_path(run_dir) if run_dir else latest_run_dir(config)
    output_dir = ensure_dir(config_path(config, "paper_outputs_dir"))
    warnings: list[str] = []
    generated_files: list[Path] = []

    if not resolved_run or not resolved_run.exists():
        warnings.append("No experiment run directory found. Run training first.")
        resolved_run = None

    comparison = collect_model_comparison(config, resolved_run) if resolved_run else []
    threat_scores = collect_threat_scores(config, resolved_run) if resolved_run else []
    experiment_config = collect_experiment_configuration(config)

    generated_files.append(write_rows_csv(output_dir / "table_model_comparison.csv", comparison, MODEL_COMPARISON_COLUMNS))
    generated_files.append(write_rows_csv(output_dir / "table_threat_scores.csv", threat_scores, THREAT_SCORE_TABLE_COLUMNS))
    generated_files.append(write_rows_csv(output_dir / "table_experiment_configuration.csv", experiment_config, EXPERIMENT_CONFIGURATION_COLUMNS))

    generated_files.append(save_system_architecture(output_dir / "figure_system_architecture.png"))
    generated_files.append(save_threat_workflow(output_dir / "figure_threat_workflow.png"))

    infer_command = "python -m src.infer --config config/config.yaml --run-dir experiments/run_<timestamp>"
    train_command = "python -m src.train --config config/config.yaml"

    y8n_rows = read_predictions(resolved_run, "yolov8n") if resolved_run else []
    y8s_rows = read_predictions(resolved_run, "yolov8s") if resolved_run else []
    path, new_warnings = save_detection_comparison(output_dir / "figure_detection_examples.png", y8n_rows, y8s_rows, infer_command)
    generated_files.append(path)
    warnings.extend(new_warnings)

    dashboard_rows = y8s_rows or y8n_rows
    path, new_warnings = save_dashboard_figure(output_dir / "figure_dashboard.png", dashboard_rows, infer_command)
    generated_files.append(path)
    warnings.extend(new_warnings)

    histories = collect_histories(config, resolved_run) if resolved_run else {str(model["name"]): [] for model in model_configs(config)}
    path, new_warnings = save_training_curves(output_dir / "figure_training_curves.png", histories, train_command)
    generated_files.append(path)
    warnings.extend(new_warnings)

    confusion_paths = {
        "yolov8n": find_confusion_matrix(resolved_run, "yolov8n") if resolved_run else None,
        "yolov8s": find_confusion_matrix(resolved_run, "yolov8s") if resolved_run else None,
    }
    path, new_warnings = save_confusion_matrix_figure(output_dir / "figure_confusion_matrix.png", confusion_paths, train_command)
    generated_files.append(path)
    warnings.extend(new_warnings)

    path, new_warnings = save_dense_swarm_example(output_dir / "figure_dense_swarm_example.png", y8s_rows, infer_command)
    generated_files.append(path)
    warnings.extend(new_warnings)

    summary_text = build_experiment_summary(config, resolved_run, comparison, threat_scores, warnings, generated_files)
    summary_path = output_dir / "experiment_summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    generated_files.append(summary_path)

    write_outputs_report(
        Path("PAPER_OUTPUTS_REPORT.md").resolve(),
        output_dir,
        resolved_run,
        generated_files,
        warnings,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures and tables from real experiment outputs.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-dir", help="Experiment run directory. Defaults to latest run.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = generate_paper_outputs(config, run_dir=args.run_dir)
    print(f"Paper outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
