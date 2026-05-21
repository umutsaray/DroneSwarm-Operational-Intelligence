from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.threat_score import BoxLike, normalize_box
from src.utils import ensure_dir

Color = tuple[int, int, int]
PUBLICATION_DPI = 300


def _save_png(image: Image.Image, path: str | Path) -> None:
    image.save(path, dpi=(PUBLICATION_DPI, PUBLICATION_DPI))


def _publication_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": PUBLICATION_DPI,
            "savefig.dpi": PUBLICATION_DPI,
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.4,
            "lines.markersize": 4.5,
        }
    )
    return plt


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Color = (25, 25, 25),
) -> None:
    lines = text.split("\n")
    line_heights = [_text_size(draw, line, font)[1] for line in lines]
    total_height = sum(line_heights) + (len(lines) - 1) * 8
    y = box[1] + ((box[3] - box[1]) - total_height) // 2
    for line, height in zip(lines, line_heights):
        width, _ = _text_size(draw, line, font)
        x = box[0] + ((box[2] - box[0]) - width) // 2
        draw.text((x, y), line, fill=fill, font=font)
        y += height + 8


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and _text_size(draw, candidate, font)[0] > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def save_placeholder_figure(
    path: str | Path,
    title: str,
    message: str,
    command: str | None = None,
    size: tuple[int, int] = (1600, 1000),
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    image = Image.new("RGB", size, (250, 250, 248))
    draw = ImageDraw.Draw(image)
    title_font = _font(44, bold=True)
    body_font = _font(28)
    mono_font = _font(24)

    draw.rectangle([40, 40, size[0] - 40, size[1] - 40], outline=(180, 180, 180), width=3)
    draw.text((80, 85), title, fill=(30, 30, 30), font=title_font)
    y = 180
    for line in _wrap_text(draw, message, body_font, size[0] - 160):
        draw.text((80, y), line, fill=(55, 55, 55), font=body_font)
        y += 42
    if command:
        y += 35
        draw.text((80, y), "Required command:", fill=(30, 30, 30), font=body_font)
        y += 50
        draw.rectangle([80, y - 10, size[0] - 80, y + 58], fill=(235, 235, 235), outline=(190, 190, 190))
        draw.text((105, y + 8), command, fill=(20, 20, 20), font=mono_font)
    _save_png(image, path)
    return path


def save_text_figure(path: str | Path, title: str, lines: list[str], size: tuple[int, int] = (1600, 1000)) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(42, bold=True)
    body_font = _font(26)
    draw.text((60, 55), title, fill=(20, 20, 20), font=title_font)
    y = 140
    for line in lines:
        for wrapped in _wrap_text(draw, line, body_font, size[0] - 120):
            draw.text((70, y), wrapped, fill=(35, 35, 35), font=body_font)
            y += 38
        y += 6
    _save_png(image, path)
    return path


def draw_boxes(
    image: Image.Image,
    boxes: list[BoxLike],
    label_prefix: str = "drone",
    color: Color = (220, 30, 30),
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, width // 500)
    label_font = _font(max(13, width // 75), bold=True)

    for index, raw_box in enumerate(boxes, start=1):
        box = normalize_box(raw_box)
        x1 = (box["x_center"] - box["width"] / 2.0) * width
        y1 = (box["y_center"] - box["height"] / 2.0) * height
        x2 = (box["x_center"] + box["width"] / 2.0) * width
        y2 = (box["y_center"] + box["height"] / 2.0) * height
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        confidence = box.get("confidence", 0.0)
        label = f"{label_prefix} {index}"
        if confidence:
            label += f" {confidence:.2f}"
        text_w, text_h = _text_size(draw, label, label_font)
        label_y = max(0, int(y1) - text_h - 6)
        draw.rectangle([x1, label_y, x1 + text_w + 8, label_y + text_h + 6], fill=(255, 255, 255))
        draw.text((x1 + 4, label_y + 3), label, fill=color, font=label_font)
    return canvas


def parse_boxes(value: str | Any) -> list[dict[str, float]]:
    if isinstance(value, list):
        return [normalize_box(item) for item in value]
    if value is None or value == "":
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [normalize_box(item) for item in parsed]


def fit_image(image: Image.Image, size: tuple[int, int], fill: Color = (245, 245, 245)) -> Image.Image:
    canvas = Image.new("RGB", size, fill)
    copy = image.convert("RGB")
    copy.thumbnail(size)
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def save_system_architecture(path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    width, height = 2200, 900
    image = Image.new("RGB", (width, height), (250, 252, 253))
    draw = ImageDraw.Draw(image)
    title_font = _font(46, bold=True)
    box_font = _font(25, bold=True)
    eq_font = _font(34, bold=True)
    draw.text((70, 55), "Figure 1. System Architecture", fill=(20, 35, 45), font=title_font)

    labels = [
        "Input\nImage/Video",
        "Image\nPreprocessing",
        "YOLOv8\nDetection",
        "Feature\nExtraction",
        "Threat Scoring\nEngine",
        "Risk\nClassification",
        "Streamlit\nDashboard",
        "Decision\nSupport",
    ]
    box_w, box_h = 225, 135
    y = 345
    x0 = 60
    gap = 42
    colors = [
        (226, 241, 249),
        (232, 245, 233),
        (255, 244, 229),
        (239, 235, 249),
        (255, 235, 238),
        (232, 245, 233),
        (225, 245, 254),
        (255, 249, 196),
    ]
    previous_right = None
    for i, label in enumerate(labels):
        x = x0 + i * (box_w + gap)
        box = (x, y, x + box_w, y + box_h)
        draw.rounded_rectangle(box, radius=18, fill=colors[i], outline=(80, 95, 105), width=3)
        _centered_text(draw, box, label, box_font)
        if previous_right is not None:
            arrow_y = y + box_h // 2
            draw.line([previous_right + 8, arrow_y, x - 14, arrow_y], fill=(60, 70, 80), width=5)
            draw.polygon([(x - 14, arrow_y - 13), (x - 14, arrow_y + 13), (x + 4, arrow_y)], fill=(60, 70, 80))
        previous_right = x + box_w

    equation = "TS = αN_norm + βD + γP"
    eq_box = (590, 610, 1605, 725)
    draw.rounded_rectangle(eq_box, radius=16, fill=(255, 255, 255), outline=(90, 90, 90), width=3)
    _centered_text(draw, eq_box, equation, eq_font)
    _save_png(image, path)
    return path


def save_threat_workflow(path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    width, height = 2100, 1250
    image = Image.new("RGB", (width, height), (250, 252, 253))
    draw = ImageDraw.Draw(image)
    title_font = _font(46, bold=True)
    step_font = _font(25, bold=True)
    small_font = _font(22)
    draw.text((70, 55), "Figure 6. Threat Scoring Workflow", fill=(20, 35, 45), font=title_font)
    steps = [
        "YOLO detections",
        "Drone count N",
        "Normalize N_norm",
        "Estimate swarm area A_swarm",
        "Compute density D = N / A_swarm",
        "Compute swarm centroid",
        "Compute proximity P = 1 - d / d_max",
        "Weighted threat score TS = αN_norm + βD + γP",
        "Risk level",
    ]
    cols = 3
    box_w, box_h = 570, 130
    x_start, y_start = 85, 185
    x_gap, y_gap = 95, 120
    positions = []
    for i, step in enumerate(steps):
        row, col = divmod(i, cols)
        x = x_start + col * (box_w + x_gap)
        y = y_start + row * (box_h + y_gap)
        positions.append((x, y, x + box_w, y + box_h))
        draw.rounded_rectangle(positions[-1], radius=18, fill=(235, 246, 252), outline=(70, 95, 115), width=3)
        _centered_text(draw, positions[-1], step, step_font)
    for a, b in zip(positions, positions[1:]):
        x1, y1 = a[2], (a[1] + a[3]) // 2
        x2, y2 = b[0], (b[1] + b[3]) // 2
        if b[0] > a[0]:
            draw.line([x1 + 8, y1, x2 - 14, y2], fill=(60, 70, 80), width=4)
            draw.polygon([(x2 - 14, y2 - 12), (x2 - 14, y2 + 12), (x2 + 4, y2)], fill=(60, 70, 80))
        else:
            mid_y = a[3] + 42
            draw.line([a[0] + box_w // 2, a[3] + 4, a[0] + box_w // 2, mid_y, b[0] + box_w // 2, mid_y, b[0] + box_w // 2, b[1] - 12], fill=(60, 70, 80), width=4)
            draw.polygon([(b[0] + box_w // 2 - 12, b[1] - 12), (b[0] + box_w // 2 + 12, b[1] - 12), (b[0] + box_w // 2, b[1] + 6)], fill=(60, 70, 80))
    note = "All terms are normalized and clipped to [0, 1]; risk levels are Low, Medium, High, and Critical."
    draw.text((95, height - 105), note, fill=(45, 45, 45), font=small_font)
    _save_png(image, path)
    return path


def _row_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def detection_panel(
    image_path: Path,
    row: dict[str, Any],
    title: str,
    size: tuple[int, int],
    boxed: bool = True,
) -> Image.Image:
    with Image.open(image_path) as source:
        boxes = parse_boxes(row.get("bounding_boxes", "")) if boxed else []
        rendered = draw_boxes(source, boxes, color=(220, 30, 30)) if boxed else source.convert("RGB")
    panel = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(panel)
    title_font = _font(25, bold=True)
    metric_font = _font(21)
    image_area = (0, 0, size[0], size[1] - 125)
    fitted = fit_image(rendered, (image_area[2], image_area[3]))
    panel.paste(fitted, (0, 0))
    draw.rectangle([0, size[1] - 125, size[0] - 1, size[1] - 1], fill=(246, 248, 249), outline=(210, 210, 210))
    draw.text((18, size[1] - 113), title, fill=(20, 20, 20), font=title_font)
    if boxed:
        metrics = (
            f"N={row.get('number_of_detections', '')} | "
            f"Conf={_row_float(row, 'average_confidence'):.3f} | "
            f"TS={_row_float(row, 'final_TS'):.3f} | "
            f"{row.get('risk_level', '')}"
        )
    else:
        metrics = image_path.name
    draw.text((18, size[1] - 65), metrics, fill=(40, 40, 40), font=metric_font)
    return panel


def select_representative_rows(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    if len(rows) < 4:
        return []
    sorted_by_count = sorted(rows, key=lambda row: (_row_float(row, "number_of_detections"), _row_float(row, "final_TS")))
    candidates = [
        ("Sparse swarm", sorted_by_count[0]),
        ("Medium swarm", sorted_by_count[len(sorted_by_count) // 2]),
        ("Dense swarm", sorted_by_count[-1]),
        ("High-threat swarm", max(rows, key=lambda row: _row_float(row, "final_TS"))),
    ]
    selected: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()
    for label, row in candidates:
        image_name = str(row.get("image_filename", ""))
        if image_name and image_name not in used:
            selected.append((label, row))
            used.add(image_name)
    for row in sorted(rows, key=lambda item: _row_float(item, "final_TS"), reverse=True):
        if len(selected) >= 4:
            break
        image_name = str(row.get("image_filename", ""))
        if image_name and image_name not in used:
            selected.append((f"Representative {len(selected) + 1}", row))
            used.add(image_name)
    return selected[:4] if len(selected) >= 4 else []


def save_detection_comparison(
    path: str | Path,
    yolov8n_rows: list[dict[str, Any]],
    yolov8s_rows: list[dict[str, Any]],
    command: str,
) -> tuple[Path, list[str]]:
    path = Path(path)
    warnings: list[str] = []
    if not yolov8n_rows or not yolov8s_rows:
        warnings.append("Detection comparison missing one or both prediction CSV files.")
        return save_placeholder_figure(
            path,
            "Figure 2. Detection Comparison Results",
            "Real YOLOv8n and YOLOv8s prediction CSV files are required to compose the 4x3 detection comparison.",
            command,
            size=(2100, 1400),
        ), warnings

    n_by_image = {str(row.get("image_filename")): row for row in yolov8n_rows}
    common_s_rows = [row for row in yolov8s_rows if str(row.get("image_filename")) in n_by_image]
    selected = select_representative_rows(common_s_rows)
    if len(selected) < 4:
        warnings.append("Detection comparison needs at least four common prediction rows.")
        return save_placeholder_figure(
            path,
            "Figure 2. Detection Comparison Results",
            "At least four common test images with YOLOv8n and YOLOv8s predictions are required.",
            command,
            size=(2100, 1400),
        ), warnings

    panel_w, panel_h = 650, 430
    header_h = 95
    left_margin = 65
    top_margin = 120
    row_gap = 30
    col_gap = 35
    width = left_margin * 2 + panel_w * 3 + col_gap * 2
    height = top_margin + header_h + len(selected) * panel_h + (len(selected) - 1) * row_gap + 65
    canvas = Image.new("RGB", (width, height), (250, 252, 253))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    header_font = _font(28, bold=True)
    row_font = _font(24, bold=True)
    draw.text((left_margin, 45), "Figure 2. Detection Comparison Results", fill=(20, 35, 45), font=title_font)
    headers = ["Original test image", "YOLOv8n prediction", "YOLOv8s prediction"]
    for col, header in enumerate(headers):
        x = left_margin + col * (panel_w + col_gap)
        draw.text((x + 12, top_margin), header, fill=(25, 25, 25), font=header_font)

    y = top_margin + header_h
    for label, s_row in selected:
        image_path = Path(str(s_row.get("image_path", "")))
        n_row = n_by_image[str(s_row.get("image_filename"))]
        if not image_path.exists():
            warnings.append(f"Missing original image for comparison: {image_path}")
            continue
        draw.text((16, y + 12), label, fill=(70, 70, 70), font=row_font)
        panels = [
            detection_panel(image_path, s_row, label, (panel_w, panel_h), boxed=False),
            detection_panel(image_path, n_row, "YOLOv8n", (panel_w, panel_h), boxed=True),
            detection_panel(image_path, s_row, "YOLOv8s", (panel_w, panel_h), boxed=True),
        ]
        for col, panel in enumerate(panels):
            x = left_margin + col * (panel_w + col_gap)
            canvas.paste(panel, (x, y))
            draw.rectangle([x, y, x + panel_w, y + panel_h], outline=(180, 185, 190), width=2)
        y += panel_h + row_gap
    _save_png(canvas, path)
    return path, warnings


def save_dashboard_figure(path: str | Path, rows: list[dict[str, Any]], command: str) -> tuple[Path, list[str]]:
    path = Path(path)
    warnings: list[str] = []
    if not rows:
        warnings.append("Dashboard figure missing prediction rows.")
        return save_placeholder_figure(
            path,
            "Figure 3. Threat Dashboard",
            "Prediction-derived rows are required to generate the static dashboard figure.",
            command,
        ), warnings
    row = max(rows, key=lambda item: _row_float(item, "final_TS"))
    image_path = Path(str(row.get("image_path", "")))
    if not image_path.exists():
        warnings.append(f"Dashboard source image missing: {image_path}")
        return save_placeholder_figure(
            path,
            "Figure 3. Threat Dashboard",
            f"The selected prediction row references a missing image: {image_path}",
            command,
        ), warnings

    width, height = 1800, 1050
    canvas = Image.new("RGB", (width, height), (250, 252, 253))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    metric_font = _font(30, bold=True)
    small_font = _font(23)
    draw.text((60, 45), "Figure 3. Prediction-Derived Threat Dashboard", fill=(20, 35, 45), font=title_font)

    with Image.open(image_path) as source:
        annotated = draw_boxes(source, parse_boxes(row.get("bounding_boxes", "")), color=(220, 30, 30))
    image_panel = fit_image(annotated, (1050, 780), fill=(235, 238, 240))
    canvas.paste(image_panel, (60, 155))
    draw.rectangle([60, 155, 1110, 935], outline=(165, 170, 175), width=3)

    metrics = [
        ("Model", str(row.get("model_name", ""))),
        ("Detected drones", str(row.get("number_of_detections", ""))),
        ("Avg confidence", f"{_row_float(row, 'average_confidence'):.3f}"),
        ("Swarm density D", f"{_row_float(row, 'D'):.3f}"),
        ("Proximity P", f"{_row_float(row, 'P'):.3f}"),
        ("Threat score TS", f"{_row_float(row, 'final_TS'):.3f}"),
        ("Risk level", str(row.get("risk_level", ""))),
    ]
    x0, y0 = 1170, 165
    card_w, card_h = 560, 86
    for i, (name, value) in enumerate(metrics):
        y = y0 + i * (card_h + 20)
        fill = (255, 255, 255)
        if name == "Risk level":
            fill = (255, 245, 235)
        draw.rounded_rectangle([x0, y, x0 + card_w, y + card_h], radius=12, fill=fill, outline=(190, 195, 200), width=2)
        draw.text((x0 + 22, y + 15), name, fill=(70, 70, 70), font=small_font)
        draw.text((x0 + 300, y + 15), value, fill=(25, 25, 25), font=metric_font)
    draw.text((60, 960), f"Source image: {image_path.name} | Values are prediction-derived from {row.get('model_name', '')}.", fill=(50, 50, 50), font=small_font)
    _save_png(canvas, path)
    return path, warnings


def _normalise_metric(values: list[float], plot_h: int, lower: float | None = None, upper: float | None = None) -> list[int]:
    if not values:
        return []
    lower = min(values) if lower is None else lower
    upper = max(values) if upper is None else upper
    if math.isclose(lower, upper):
        return [plot_h // 2 for _ in values]
    return [int(plot_h - ((v - lower) / (upper - lower)) * plot_h) for v in values]


def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: dict[str, list[float]],
    colors: dict[str, Color],
) -> None:
    title_font = _font(22, bold=True)
    label_font = _font(17)
    x1, y1, x2, y2 = box
    draw.rectangle(box, fill="white", outline=(185, 190, 195), width=2)
    draw.text((x1 + 16, y1 + 12), title, fill=(25, 25, 25), font=title_font)
    plot = (x1 + 55, y1 + 55, x2 - 30, y2 - 45)
    draw.line([plot[0], plot[3], plot[2], plot[3]], fill=(150, 150, 150), width=2)
    draw.line([plot[0], plot[1], plot[0], plot[3]], fill=(150, 150, 150), width=2)

    all_values = [v for values in series.values() for v in values]
    if not all_values:
        draw.text((plot[0] + 20, plot[1] + 40), "No real results.csv values available.", fill=(120, 60, 60), font=label_font)
        return
    min_v, max_v = min(all_values), max(all_values)
    draw.text((plot[0], plot[1] - 22), f"{max_v:.3f}", fill=(80, 80, 80), font=label_font)
    draw.text((plot[0], plot[3] + 8), f"{min_v:.3f}", fill=(80, 80, 80), font=label_font)

    for name, values in series.items():
        if len(values) < 2:
            continue
        points = []
        ys = _normalise_metric(values, plot[3] - plot[1], min_v, max_v)
        for index, y_offset in enumerate(ys):
            x = plot[0] + int(index / (len(values) - 1) * (plot[2] - plot[0]))
            y = plot[1] + y_offset
            points.append((x, y))
        draw.line(points, fill=colors[name], width=4)
    legend_x = x1 + 18
    legend_y = y2 - 31
    for name, color in colors.items():
        draw.line([legend_x, legend_y + 8, legend_x + 30, legend_y + 8], fill=color, width=4)
        draw.text((legend_x + 38, legend_y), name, fill=(45, 45, 45), font=label_font)
        legend_x += 150


def save_training_curves(path: str | Path, histories: dict[str, list[dict[str, Any]]], command: str) -> tuple[Path, list[str]]:
    path = Path(path)
    ensure_dir(path.parent)
    warnings: list[str] = []
    available = {name: rows for name, rows in histories.items() if rows}
    if not available:
        warnings.append("Training curves missing results.csv files.")
        return save_placeholder_figure(
            path,
            "Figure 4. Training Curves",
            "Real YOLO results.csv files are required to plot training curves.",
            command,
            size=(1900, 1200),
        ), warnings

    metrics = [
        ("Train Box Loss", "train/box_loss", "Loss"),
        ("Validation Box Loss", "val/box_loss", "Loss"),
        ("Precision", "metrics/precision(B)", "Precision"),
        ("Recall", "metrics/recall(B)", "Recall"),
        ("mAP50", "metrics/mAP50(B)", "mAP50"),
        ("mAP50-95", "metrics/mAP50-95(B)", "mAP50-95"),
    ]

    plt = _publication_pyplot()
    colors = {"yolov8n": "#1f77b4", "yolov8s": "#d95f02"}
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 12.2), dpi=PUBLICATION_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle("Figure 4. YOLOv8 Training Curves", fontsize=20, fontweight="bold", y=0.995)

    for axis, (title, column, ylabel) in zip(axes.flat, metrics):
        plotted = False
        for model_name, rows in histories.items():
            x_values: list[int] = []
            y_values: list[float] = []
            for epoch_index, row_data in enumerate(rows, start=1):
                try:
                    value = float(row_data.get(column, "") or "")
                except (TypeError, ValueError):
                    continue
                x_values.append(epoch_index)
                y_values.append(value)
            if y_values:
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    markevery=max(1, len(y_values) // 12),
                    color=colors.get(model_name, "#555555"),
                    label=model_name,
                )
                plotted = True

        axis.set_title(title, pad=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.45)
        axis.tick_params(axis="both", which="major", width=1.1, length=5)
        for spine in axis.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color("#444444")
        if plotted:
            axis.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
        else:
            axis.text(
                0.5,
                0.5,
                f"No real values for\n{column}",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#8a3b2c",
                fontsize=13,
            )

    missing = [name for name, rows in histories.items() if not rows]
    if missing:
        warnings.append(f"Missing training results for: {', '.join(missing)}")
        fig.text(
            0.5,
            0.01,
            f"Missing training results for: {', '.join(missing)}",
            ha="center",
            color="#8a3b2c",
            fontsize=12,
        )

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.965), h_pad=2.0, w_pad=1.6)
    fig.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path, warnings


def save_confusion_matrix_figure(path: str | Path, image_paths: dict[str, Path | None], command: str) -> tuple[Path, list[str]]:
    path = Path(path)
    ensure_dir(path.parent)
    warnings: list[str] = []
    if not any(p and p.exists() for p in image_paths.values()):
        warnings.append("Confusion matrix images are missing.")
        return save_placeholder_figure(
            path,
            "Figure 5. Confusion Matrix",
            "Real confusion matrix images are required for YOLOv8n and YOLOv8s.",
            command,
            size=(1800, 1000),
        ), warnings

    plt = _publication_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.2), dpi=PUBLICATION_DPI)
    fig.patch.set_facecolor("white")
    fig.suptitle("Figure 5. Confusion Matrix", fontsize=20, fontweight="bold", y=0.985)
    for axis, model_name in zip(axes, ["yolov8n", "yolov8s"]):
        axis.set_title(model_name, pad=12)
        axis.axis("off")
        source = image_paths.get(model_name)
        if source and source.exists():
            with Image.open(source) as img:
                axis.imshow(img.convert("RGB"))
        else:
            warnings.append(f"Missing confusion matrix for {model_name}")
            axis.text(
                0.5,
                0.5,
                "Missing\nRun training and validation first.",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#8a3b2c",
                fontsize=16,
                fontweight="bold",
            )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94), w_pad=1.5)
    fig.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path, warnings


def save_dense_swarm_example(path: str | Path, rows: list[dict[str, Any]], command: str) -> tuple[Path, list[str]]:
    path = Path(path)
    warnings: list[str] = []
    if not rows:
        warnings.append("Dense swarm example missing YOLOv8s prediction rows.")
        return save_placeholder_figure(
            path,
            "Figure 7. Dense Swarm Detection Example",
            "YOLOv8s prediction rows are required to select the highest-count or highest-threat example.",
            command,
        ), warnings
    row = max(rows, key=lambda item: (_row_float(item, "number_of_detections"), _row_float(item, "final_TS")))
    image_path = Path(str(row.get("image_path", "")))
    if not image_path.exists():
        warnings.append(f"Dense swarm source image missing: {image_path}")
        return save_placeholder_figure(
            path,
            "Figure 7. Dense Swarm Detection Example",
            f"The selected prediction row references a missing image: {image_path}",
            command,
        ), warnings

    width, height = 1900, 1000
    canvas = Image.new("RGB", (width, height), (250, 252, 253))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    metric_font = _font(29, bold=True)
    small_font = _font(23)
    draw.text((60, 45), "Figure 7. Dense Swarm Detection Example", fill=(20, 35, 45), font=title_font)
    with Image.open(image_path) as source:
        original = fit_image(source, (680, 620), fill=(235, 238, 240))
        annotated = fit_image(draw_boxes(source, parse_boxes(row.get("bounding_boxes", "")), color=(220, 30, 30)), (680, 620), fill=(235, 238, 240))
    draw.text((95, 135), "Original image", fill=(30, 30, 30), font=_font(28, bold=True))
    draw.text((820, 135), "YOLOv8s detection output", fill=(30, 30, 30), font=_font(28, bold=True))
    canvas.paste(original, (60, 180))
    canvas.paste(annotated, (785, 180))
    draw.rectangle([60, 180, 740, 800], outline=(165, 170, 175), width=3)
    draw.rectangle([785, 180, 1465, 800], outline=(165, 170, 175), width=3)
    x0, y0 = 1510, 190
    metrics = [
        ("Model", "YOLOv8s"),
        ("Drone count", str(row.get("number_of_detections", ""))),
        ("Avg confidence", f"{_row_float(row, 'average_confidence'):.3f}"),
        ("Threat score", f"{_row_float(row, 'final_TS'):.3f}"),
        ("Risk level", str(row.get("risk_level", ""))),
    ]
    for index, (name, value) in enumerate(metrics):
        y = y0 + index * 105
        draw.rounded_rectangle([x0, y, x0 + 330, y + 82], radius=12, fill="white", outline=(190, 195, 200), width=2)
        draw.text((x0 + 18, y + 12), name, fill=(70, 70, 70), font=small_font)
        draw.text((x0 + 18, y + 42), value, fill=(20, 20, 20), font=metric_font)
    draw.text((60, 845), f"Selected from real YOLOv8s predictions by highest detection count, then highest TS. Image: {image_path.name}", fill=(45, 45, 45), font=small_font)
    _save_png(canvas, path)
    return path, warnings
