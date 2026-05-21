# Experiment Summary

This folder preserves publication figures and tables generated from the completed YOLOv8n/YOLOv8s experiment run used during repository preparation.

## Dataset

- Full experiment dataset size: 117 images
- Train images: 81
- Validation images: 17
- Test images: 19
- Augmented train images: 324
- Total training images after augmentation: 405
- Academic note: The training set was expanded through augmentation, while validation and test sets were kept unchanged.

The public GitHub repository includes only a lightweight sample dataset for software validation. The sample dataset must not be interpreted as the full experimental dataset.

## Trained Models

- Models with available metrics: yolov8n, yolov8s
- Best model by available mAP metrics: yolov8s

## Performance Comparison

- yolov8n: Precision=0.93372, Recall=0.86964, mAP50=0.86592, mAP50-95=0.51773, FPS mean=6.760593
- yolov8s: Precision=0.98642, Recall=0.89689, mAP50=0.91157, mAP50-95=0.54696, FPS mean=3.254703

## Inference Summary

- Prediction-derived threat rows: 38
- Highest threat image: drone_swarm_0114.png (yolov8s), TS=0.768911, risk=Critical

## Generated Figures and Tables

- `paper_outputs/table_model_comparison.csv`
- `paper_outputs/table_threat_scores.csv`
- `paper_outputs/table_experiment_configuration.csv`
- `paper_outputs/table_realtime_inference.csv`
- `paper_outputs/table_realtime_inference_gpu.csv`
- `paper_outputs/figure_system_architecture.png`
- `paper_outputs/figure_threat_workflow.png`
- `paper_outputs/figure_detection_examples.png`
- `paper_outputs/figure_dashboard.png`
- `paper_outputs/figure_training_curves.png`
- `paper_outputs/figure_confusion_matrix.png`
- `paper_outputs/figure_dense_swarm_example.png`

## Missing Data Warnings

- None for the archived publication outputs.

## Remaining Limitations

- Threat-score weights and thresholds still require domain validation.
- Image-normalized distance and area are not physical-world measurements.
- Missing artifacts are never replaced with simulated values.
- Full dataset and trained weights are available upon reasonable academic request.
