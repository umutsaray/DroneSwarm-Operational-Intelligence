from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image

from src.infer import model_weight_path, test_image_paths
from src.utils import config_path, ensure_dir, load_config, model_configs

OUTPUT_COLUMNS = [
    "Model",
    "Average_Inference_Time_ms",
    "Estimated_FPS",
    "Device",
    "CUDA_Available",
    "GPU",
    "Input_Resolution",
    "Warmup_Runs",
    "Test_Images",
    "Weights",
]

MINIMAL_OUTPUT_COLUMNS = [
    "Model",
    "Device",
    "Average inference time (ms)",
    "Estimated FPS",
]


def estimated_fps(avg_inference_time_ms: float) -> float:
    return 1000.0 / avg_inference_time_ms if avg_inference_time_ms > 0 else 0.0


def gpu_name_from_system() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "Not detected"
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ", ".join(names) if names else "Not detected"


def torch_device(requested_device: str = "auto") -> tuple[str, bool, str]:
    try:
        import torch
    except ImportError:
        return "cpu", False, "PyTorch not installed"
    cuda_available = bool(torch.cuda.is_available())
    if requested_device.startswith("cuda") and not cuda_available:
        return requested_device, False, gpu_name_from_system()
    if requested_device.startswith("cuda") and cuda_available:
        return requested_device, True, torch.cuda.get_device_name(0)
    if requested_device == "auto" and cuda_available:
        return "0", True, torch.cuda.get_device_name(0)
    return "cpu", False, gpu_name_from_system()


def synchronize_if_cuda(cuda_available: bool) -> None:
    if not cuda_available:
        return
    import torch

    torch.cuda.synchronize()


def load_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def benchmark_model(
    model_name: str,
    weights: Path,
    images: list[Image.Image],
    device: str,
    cuda_available: bool,
    imgsz: int,
    warmup_runs: int,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    warmup_image = images[0]
    for _ in range(warmup_runs):
        model.predict(warmup_image, imgsz=imgsz, device=device, verbose=False)
    synchronize_if_cuda(cuda_available)

    timings_ms: list[float] = []
    for image in images:
        synchronize_if_cuda(cuda_available)
        start = time.perf_counter()
        model.predict(image, imgsz=imgsz, device=device, verbose=False)
        synchronize_if_cuda(cuda_available)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

    avg_ms = sum(timings_ms) / len(timings_ms)
    return {
        "Model": model_name,
        "Average_Inference_Time_ms": round(avg_ms, 3),
        "Estimated_FPS": round(estimated_fps(avg_ms), 3),
        "Device": "cuda:0" if cuda_available and device != "cpu" else "cpu",
        "Input_Resolution": imgsz,
        "Warmup_Runs": warmup_runs,
        "Test_Images": len(images),
        "Weights": str(weights),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    cuda_available: bool,
    gpu_name: str,
    device: str,
    image_count: int,
    imgsz: int,
    warmup_runs: int,
) -> None:
    ensure_dir(path.parent)
    lines = [
        "# Real-Time Inference Benchmark Summary",
        "",
        f"- Hardware platform: {platform.platform()}",
        f"- GPU detected by system: {gpu_name}",
        f"- PyTorch CUDA available: {cuda_available}",
        f"- Measured device: {'cuda:0' if cuda_available and device != 'cpu' else 'cpu'}",
        f"- Input resolution: {imgsz}x{imgsz}",
        f"- Warmup inferences per model: {warmup_runs}",
        f"- Timed test images per model: {image_count}",
        "",
        "## Results",
        "",
        "| Model | Average inference time (ms) | Estimated FPS |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Model']} | {row['Average_Inference_Time_ms']} | {row['Estimated_FPS']} |"
        )
    lines.extend(
        [
            "",
            "## Measurement Note",
            "",
            "FPS is estimated as `1000 / average_inference_time_ms`.",
        ]
    )
    if not cuda_available:
        lines.append(
            "An NVIDIA GPU may be present, but this Python environment does not expose CUDA through PyTorch. "
            "The values above are CPU measurements, not RTX GPU measurements. Install a CUDA-enabled PyTorch build to benchmark RTX 3050 Laptop GPU deployment."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gpu_unavailable_outputs(output_dir: Path, gpu_name: str, requested_device: str) -> None:
    write_csv(output_dir / "table_realtime_inference_gpu.csv", [], MINIMAL_OUTPUT_COLUMNS)
    lines = [
        "# GPU Real-Time Inference Benchmark Summary",
        "",
        f"- Requested device: {requested_device}",
        f"- GPU detected by system: {gpu_name}",
        "- PyTorch CUDA available: False",
        "",
        "GPU benchmark was not executed because CUDA is not available to PyTorch in this Python environment.",
        "No latency or FPS values were fabricated.",
        "",
        "Attempted CUDA install command:",
        "",
        "```bash",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "```",
        "",
        "The current Python environment did not have a compatible CUDA 12.1 PyTorch wheel.",
    ]
    (output_dir / "realtime_inference_gpu_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    config: dict[str, Any],
    warmup_runs: int = 10,
    imgsz: int = 640,
    requested_device: str = "auto",
    output_suffix: str = "",
) -> list[dict[str, Any]]:
    try:
        import ultralytics  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run pip install -r requirements.txt first.") from exc

    image_paths = test_image_paths(config)
    if not image_paths:
        raise ValueError("No test images found. Run python -m src.dataset --config config/config.yaml first.")
    images = load_images(image_paths)
    device, cuda_available, gpu_name = torch_device(requested_device)
    output_dir = ensure_dir(config_path(config, "paper_outputs_dir"))
    if requested_device.startswith("cuda") and not cuda_available:
        write_gpu_unavailable_outputs(output_dir, gpu_name, requested_device)
        raise RuntimeError(
            f"Requested {requested_device}, but torch.cuda.is_available() is False. GPU benchmark was not executed."
        )

    rows: list[dict[str, Any]] = []
    for model_cfg in model_configs(config):
        model_name = str(model_cfg["name"])
        weights = model_weight_path(config, model_name)
        row = benchmark_model(
            model_name=model_name,
            weights=weights,
            images=images,
            device=device,
            cuda_available=cuda_available,
            imgsz=imgsz,
            warmup_runs=warmup_runs,
        )
        row["CUDA_Available"] = cuda_available
        row["GPU"] = gpu_name
        rows.append(row)

    csv_name = f"table_realtime_inference{'_' + output_suffix if output_suffix else ''}.csv"
    summary_name = f"realtime_inference{'_' + output_suffix if output_suffix else ''}_summary.md"
    if output_suffix == "gpu":
        minimal_rows = [
            {
                "Model": row["Model"],
                "Device": row["Device"],
                "Average inference time (ms)": row["Average_Inference_Time_ms"],
                "Estimated FPS": row["Estimated_FPS"],
            }
            for row in rows
        ]
        write_csv(output_dir / csv_name, minimal_rows, MINIMAL_OUTPUT_COLUMNS)
    else:
        write_csv(output_dir / csv_name, rows, OUTPUT_COLUMNS)
    write_summary(
        output_dir / summary_name,
        rows=rows,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        device=device,
        image_count=len(images),
        imgsz=imgsz,
        warmup_runs=warmup_runs,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 real-time inference latency on the test set.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--output-suffix", default="", help="Use gpu to write table_realtime_inference_gpu.csv and realtime_inference_gpu_summary.md")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = run_benchmark(
        config,
        warmup_runs=args.warmup_runs,
        imgsz=args.imgsz,
        requested_device=args.device,
        output_suffix=args.output_suffix,
    )
    for row in rows:
        print(
            f"{row['Model']}: {row['Average_Inference_Time_ms']} ms/image, "
            f"{row['Estimated_FPS']} FPS ({row['Device']})"
        )


if __name__ == "__main__":
    main()
