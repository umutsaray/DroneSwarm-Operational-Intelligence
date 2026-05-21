# Real-Time Inference Benchmark Summary

- Hardware platform: Windows-11-10.0.26200-SP0
- GPU detected by system: NVIDIA GeForce RTX 3050 Laptop GPU
- PyTorch CUDA available: False
- Measured device: cpu
- Input resolution: 640x640
- Warmup inferences per model: 10
- Timed test images per model: 19

## Results

| Model | Average inference time (ms) | Estimated FPS |
|---|---:|---:|
| yolov8n | 40.241 | 24.85 |
| yolov8s | 93.164 | 10.734 |

## Measurement Note

FPS is estimated as `1000 / average_inference_time_ms`.
An NVIDIA GPU may be present, but this Python environment does not expose CUDA through PyTorch. The values above are CPU measurements, not RTX GPU measurements. Install a CUDA-enabled PyTorch build to benchmark RTX 3050 Laptop GPU deployment.
