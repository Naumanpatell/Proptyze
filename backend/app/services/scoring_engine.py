import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Max penalty points per defect class (deducted from 100)
_CONDITION_PENALTIES: dict[str, float] = {
    "mould":          20.0,  # health hazard — highest priority
    "wall_crack":     15.0,  # structural concern
    "damp":           15.0,  # leads to mould and structural damage
    "broken_fixture": 10.0,
    "peeling_paint":   5.0,  # cosmetic only
}

_SECURITY_PENALTIES: dict[str, float] = {
    "weak_entry":        20.0,  # direct break-in risk
    "fence_gap":         10.0,
    "camera_blind_spot": 10.0,
}

_ALL_PENALTIES = {**_CONDITION_PENALTIES, **_SECURITY_PENALTIES}

# How many frames a defect appears in affects? how bad it is?, but with diminishing returns
def _freq_multiplier(count: int) -> float:
    if count == 1:
        return 1.0
    if count <= 4:
        return 1.2
    return 1.4


def _detection_penalty(detections: list[dict]) -> float:
    by_class: dict[str, list[float]] = {}
    for d in detections:
        by_class.setdefault(d["class"], []).append(d["confidence"])

    total = 0.0
    for cls, confidences in by_class.items():
        base = _ALL_PENALTIES.get(cls, 5.0)
        freq_mult = _freq_multiplier(len(confidences))
        mean_conf = sum(confidences) / len(confidences)
        # Map confidence [0.25, 1.0] → weight [0.5, 1.0] so borderline detections hurt less
        conf_weight = max(0.5, min(1.0, 0.5 + 0.5 * (mean_conf - 0.25) / 0.75))
        total += base * freq_mult * conf_weight
        logger.debug("  %s × %d frames → penalty %.1f", cls, len(confidences), base * freq_mult * conf_weight)

    return total


def _lighting_penalty(frame_paths: list[str]) -> float:
    """Penalise properties where most frames are poorly lit."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return 0.0

    brightnesses = []
    for path in frame_paths[:20]:  # sample up to 20 frames
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            brightnesses.append(float(np.mean(img)))

    if not brightnesses:
        return 0.0

    mean_brightness = sum(brightnesses) / len(brightnesses)
    if mean_brightness < 60:
        return 10.0  # consistently dark — poor lighting
    if mean_brightness < 80:
        return 5.0   # dim but not critical
    return 0.0


def _grade(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


def compute_score(
    detections: list[dict],
    neighbourhood_data: dict,
    frame_paths: Optional[list[str]] = None,
) -> dict:
    """
    Returns:
        score (int): 0–100, higher is better
        grade (str): A / B / C / D
        breakdown (dict): penalty components for the report generator
    """
    det_penalty = _detection_penalty(detections)
    light_penalty = _lighting_penalty(frame_paths) if frame_paths else 0.0
    total_penalty = det_penalty + light_penalty

    score = max(0, min(100, round(100 - total_penalty)))
    grade = _grade(score)

    logger.info(
        "Score %d (%s) | detection_penalty=%.1f lighting_penalty=%.1f",
        score, grade, det_penalty, light_penalty,
    )

    classes_found = list({d["class"] for d in detections})

    return {
        "score": score,
        "grade": grade,
        "breakdown": {
            "detection_penalty": round(det_penalty, 1),
            "lighting_penalty":  round(light_penalty, 1),
            "classes_found":     classes_found,
        },
    }
