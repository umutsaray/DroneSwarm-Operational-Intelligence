from __future__ import annotations

import math
from typing import Any, Iterable

BoxLike = dict[str, Any] | tuple[int, float, float, float, float]


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def normalize_box(box: BoxLike) -> dict[str, float]:
    if isinstance(box, dict):
        return {
            "class_id": int(box.get("class_id", box.get("class", 0))),
            "x_center": float(box.get("x_center", box.get("xc", 0.0))),
            "y_center": float(box.get("y_center", box.get("yc", 0.0))),
            "width": float(box.get("width", box.get("w", 0.0))),
            "height": float(box.get("height", box.get("h", 0.0))),
            "confidence": float(box.get("confidence", 0.0)),
        }

    cls, xc, yc, width, height = box
    return {
        "class_id": int(cls),
        "x_center": float(xc),
        "y_center": float(yc),
        "width": float(width),
        "height": float(height),
        "confidence": 0.0,
    }


def normalized_boxes(boxes: Iterable[BoxLike]) -> list[dict[str, float]]:
    return [normalize_box(box) for box in boxes]


def box_to_xyxy(box: dict[str, float]) -> tuple[float, float, float, float]:
    half_w = box["width"] / 2.0
    half_h = box["height"] / 2.0
    x1 = clamp(box["x_center"] - half_w)
    y1 = clamp(box["y_center"] - half_h)
    x2 = clamp(box["x_center"] + half_w)
    y2 = clamp(box["y_center"] + half_h)
    return x1, y1, x2, y2


def calculate_a_swarm(boxes: Iterable[BoxLike]) -> float:
    """Minimum normalized rectangle area covering all detected drone boxes."""
    parsed = normalized_boxes(boxes)
    if not parsed:
        return 0.0

    coords = [box_to_xyxy(box) for box in parsed]
    min_x = min(x1 for x1, _, _, _ in coords)
    min_y = min(y1 for _, y1, _, _ in coords)
    max_x = max(x2 for _, _, x2, _ in coords)
    max_y = max(y2 for _, _, _, y2 in coords)
    return clamp((max_x - min_x) * (max_y - min_y), 0.0, 1.0)


def calculate_centroid(boxes: Iterable[BoxLike]) -> tuple[float, float] | None:
    parsed = normalized_boxes(boxes)
    if not parsed:
        return None
    return (
        sum(box["x_center"] for box in parsed) / len(parsed),
        sum(box["y_center"] for box in parsed) / len(parsed),
    )


def calculate_proximity(
    boxes: Iterable[BoxLike],
    protected_region_center: list[float] | tuple[float, float],
    d_max: float,
) -> tuple[float, float | None]:
    """Return P and d, where P = max(0, min(1, 1 - d / d_max))."""
    centroid = calculate_centroid(boxes)
    if centroid is None:
        return 0.0, None
    d = math.hypot(
        centroid[0] - float(protected_region_center[0]),
        centroid[1] - float(protected_region_center[1]),
    )
    if d_max <= 0:
        raise ValueError("d_max must be positive")
    return clamp(1.0 - d / d_max), d


def risk_level(threat_score: float) -> str:
    score = clamp(threat_score)
    if score < 0.25:
        return "Low"
    if score < 0.50:
        return "Medium"
    if score < 0.75:
        return "High"
    return "Critical"


def calculate_threat_score(boxes: Iterable[BoxLike], config: dict[str, Any]) -> dict[str, Any]:
    """Implement the paper formula exactly:

    TS = alpha * N_norm + beta * D + gamma * P
    D = min((N / A_swarm) / D_max, 1)
    P = max(0, min(1, 1 - d / d_max))
    """
    score_cfg = config.get("threat_score", config)
    parsed = normalized_boxes(boxes)
    n = len(parsed)

    alpha = float(score_cfg["alpha"])
    beta = float(score_cfg["beta"])
    gamma = float(score_cfg["gamma"])
    n_max = float(score_cfg["N_max"])
    density_max = float(score_cfg["D_max"])
    d_max = float(score_cfg["d_max"])
    center = score_cfg.get("protected_region_center", [0.5, 0.5])
    min_area = float(score_cfg.get("min_swarm_area", 1e-6))

    if n_max <= 0:
        raise ValueError("N_max must be positive")
    if density_max <= 0:
        raise ValueError("D_max must be positive")
    if min_area <= 0:
        raise ValueError("min_swarm_area must be positive")

    n_norm = clamp(n / n_max)
    a_swarm = calculate_a_swarm(parsed)
    effective_area = max(a_swarm, min_area) if n else min_area
    density_raw = (n / effective_area) if n else 0.0
    density = clamp(density_raw / density_max)
    proximity, distance = calculate_proximity(parsed, center, d_max)
    ts = clamp(alpha * n_norm + beta * density + gamma * proximity)

    centroid = calculate_centroid(parsed)
    return {
        "N": n,
        "drone_count": n,
        "N_norm": round(n_norm, 6),
        "A_swarm": round(a_swarm, 8),
        "D_raw": round(density_raw, 6),
        "D": round(density, 6),
        "d": None if distance is None else round(distance, 6),
        "d_max": d_max,
        "P": round(proximity, 6),
        "TS": round(ts, 6),
        "final_TS": round(ts, 6),
        "risk_level": risk_level(ts),
        "centroid_x": None if centroid is None else round(centroid[0], 6),
        "centroid_y": None if centroid is None else round(centroid[1], 6),
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "N_max": n_max,
        "D_max": density_max,
    }
