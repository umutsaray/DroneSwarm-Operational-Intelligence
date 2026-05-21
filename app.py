from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.evaluate import collect_model_comparison
from src.infer import model_weight_path, predict_image
from src.utils import config_path, get_model_config, latest_run_dir, load_config, model_configs
from src.visualization import draw_boxes


st.set_page_config(page_title="Drone Swarm Threat Assessment", layout="wide")
st.title("Drone Swarm Threat Assessment System")
st.caption("Experiment-driven YOLOv8 detection and paper-formula threat scoring.")


@st.cache_data
def cached_config() -> dict:
    return load_config("config/config.yaml")


@st.cache_resource
def load_yolo_model(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


def prediction_source_note() -> None:
    st.info("Displayed inference values are prediction-derived from a YOLO model and the configured paper formula.")


def show_single_image(config: dict) -> None:
    st.subheader("Single Image Inference")
    model_names = [model["name"] for model in model_configs(config)]
    model_name = st.selectbox("Model", model_names, help="Uses models/<model>/best.pt when available.")
    model_cfg = get_model_config(config, model_name)

    try:
        weights = model_weight_path(config, model_name)
        st.caption(f"Model weights: `{weights}`")
    except FileNotFoundError as exc:
        weights = None
        st.warning(f"No trained weights found for `{model_name}`. Train the model first. {exc}")

    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.stop()
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if weights is None:
        st.stop()

    if st.button("Run prediction-derived inference", type="primary"):
        try:
            model = load_yolo_model(str(weights))
            boxes, metrics, elapsed = predict_image(
                model=model,
                image=image,
                model_name=model_name,
                class_ids=list(model_cfg.get("drone_class_ids", [0])),
                score_config=config,
                confidence_threshold=float(config["inference"]["confidence_threshold"]),
                iou_threshold=float(config["inference"]["iou_threshold"]),
            )
        except ImportError:
            st.error("Ultralytics is not installed. Run `pip install -r requirements.txt`.")
            st.stop()
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.stop()

        prediction_source_note()
        annotated = draw_boxes(image, boxes)
        left, right = st.columns([2, 1])
        with left:
            st.image(annotated, caption="Prediction-derived detections", use_container_width=True)
        with right:
            avg_conf = sum(float(box["confidence"]) for box in boxes) / len(boxes) if boxes else 0.0
            st.metric("Detected drones", len(boxes))
            st.metric("Average confidence", round(avg_conf, 4))
            st.metric("Threat score TS", metrics["final_TS"])
            st.metric("Risk level", metrics["risk_level"])
            st.metric("Inference time (s)", round(elapsed, 4))
            st.json(metrics)


def show_experiment_viewer(config: dict) -> None:
    st.subheader("Dataset Experiment Viewer")
    experiments_dir = config_path(config, "experiments_dir")
    runs = sorted(experiments_dir.glob("run_*"), key=lambda p: p.name, reverse=True)
    if not runs:
        st.warning("No experiment runs found. Run training and inference first.")
        return

    default_run = latest_run_dir(config)
    index = runs.index(default_run) if default_run in runs else 0
    run_dir = st.selectbox("Experiment run", runs, index=index, format_func=lambda p: p.name)
    st.caption(f"Experiment path: `{run_dir}`")

    comparison = collect_model_comparison(config, run_dir)
    st.markdown("### Model Comparison")
    st.caption("Training metrics come from YOLO results.csv. Detection/threat columns are prediction-derived when prediction CSV files exist.")
    st.dataframe(pd.DataFrame(comparison), use_container_width=True)

    st.markdown("### Prediction CSV Files")
    prediction_files = sorted((run_dir / "predictions").glob("predictions_*.csv"))
    if not prediction_files:
        st.info("No prediction CSV files found for this run. Run `python -m src.infer --run-dir <run>`.")
    for csv_path in prediction_files:
        with st.expander(csv_path.name, expanded=True):
            st.caption("Prediction-derived rows. Bounding boxes are JSON-encoded normalized YOLO boxes.")
            st.dataframe(pd.read_csv(csv_path), use_container_width=True)

    st.markdown("### Saved Figures")
    figure_files = sorted((run_dir / "figures").glob("*.png"))
    if not figure_files:
        st.info("No saved figures found for this run yet.")
    for figure in figure_files:
        with st.expander(figure.name):
            st.image(str(figure), use_container_width=True)


def show_paper_outputs(config: dict) -> None:
    st.subheader("Paper Outputs")
    output_dir = config_path(config, "paper_outputs_dir")
    st.caption(f"Output path: `{output_dir}`")
    ordered = [
        "figure_system_architecture.png",
        "figure_detection_examples.png",
        "figure_dashboard.png",
        "figure_training_curves.png",
        "figure_confusion_matrix.png",
        "figure_threat_workflow.png",
        "figure_dense_swarm_example.png",
        "table_model_comparison.csv",
        "table_threat_scores.csv",
        "table_realtime_inference.csv",
        "table_realtime_inference_gpu.csv",
        "table_experiment_configuration.csv",
        "experiment_summary.md",
        "realtime_inference_summary.md",
        "realtime_inference_gpu_summary.md",
    ]
    if not output_dir.exists():
        st.warning("No paper outputs found. Run `python -m src.evaluate` after training/inference.")
        return

    for name in ordered:
        path = output_dir / name
        if not path.exists():
            st.warning(f"Missing `{name}`. Run `python -m src.evaluate --config config/config.yaml --run-dir experiments/run_<timestamp>`.")
            continue
        if path.suffix.lower() == ".csv":
            with st.expander(path.name, expanded=path.name == "table_model_comparison.csv"):
                st.dataframe(pd.read_csv(path), use_container_width=True)
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with st.expander(path.name):
                st.image(str(path), use_container_width=True)
        elif path.suffix.lower() == ".md":
            with st.expander(path.name):
                st.markdown(path.read_text(encoding="utf-8"))


def main() -> None:
    config = cached_config()
    with st.sidebar:
        st.header("Workflow")
        mode = st.radio(
            "View",
            ["Single image inference", "Dataset experiment viewer", "Paper outputs"],
        )
        st.markdown("---")
        st.caption("No random/demo metrics are generated. Missing outputs are shown as missing.")

    if mode == "Single image inference":
        show_single_image(config)
    elif mode == "Dataset experiment viewer":
        show_experiment_viewer(config)
    else:
        show_paper_outputs(config)


if __name__ == "__main__":
    main()
