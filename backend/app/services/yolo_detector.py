import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ML_DIR = Path(__file__).resolve().parents[3] / "ml"

_WEIGHT_PATHS = {
    "condition": _ML_DIR / "runs" / "condition" / "v1" / "weights" / "best.pt",
    "security":  _ML_DIR / "runs" / "security"  / "v1" / "weights" / "best.pt",
}

_CONF_THRESHOLD = 0.25

# Cached model instances to avoid reloading on every inference call
_models: dict = {}


def _load_model(model_name: str):
    if model_name in _models:
        return _models[model_name]

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed — run: pip install ultralytics")

    weights = _WEIGHT_PATHS[model_name]
    if not weights.exists():
        raise FileNotFoundError(
            f"No weights for '{model_name}' at {weights}\n"
            f"Run: python backend/ml/train.py --model {model_name}"
        )

    logger.info("Loading %s model from %s", model_name, weights)
    _models[model_name] = YOLO(str(weights))
    return _models[model_name]


def _run_model(model_name: str, frame_paths: list[str]) -> list[dict]:
    model = _load_model(model_name)
    results = model.predict(
        source=frame_paths,
        conf=_CONF_THRESHOLD,
        verbose=False,
        stream=False,
    )

    detections = []
    for result, frame_path in zip(results, frame_paths):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        names = result.names
        for box in boxes:
            cls_idx = int(box.cls[0])
            detections.append({
                "frame":      frame_path,
                "model":      model_name,
                "class":      names[cls_idx],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {
                    "x1": round(float(box.xyxy[0][0]), 1),
                    "y1": round(float(box.xyxy[0][1]), 1),
                    "x2": round(float(box.xyxy[0][2]), 1),
                    "y2": round(float(box.xyxy[0][3]), 1),
                },
            })

    return detections


def detect(
    frame_paths: list[str],
    models: Optional[list[str]] = None,
) -> list[dict]:
  
    if not frame_paths:
        return []

    active_models = models or ["condition", "security"]
    all_detections: list[dict] = []

    for model_name in active_models:
        try:
            found = _run_model(model_name, frame_paths)
            logger.info("[%s] %d detections across %d frames", model_name, len(found), len(frame_paths))
            all_detections.extend(found)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s model — weights not found: %s", model_name, exc)

    return all_detections