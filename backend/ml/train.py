import argparse
from pathlib import Path

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.")
    print("       Run: pip install ultralytics")
    raise SystemExit(1)

ML_DIR = Path(__file__).parent

MODELS = {
    "condition": {
        "data_yaml": ML_DIR / "datasets" / "condition" / "merged" / "data.yaml",
        "runs_dir":  ML_DIR / "runs" / "condition",
        "classes":   ["wall_crack", "damp", "mould", "peeling_paint", "broken_fixture"],
    },
    "security": {
        "data_yaml": ML_DIR / "datasets" / "security" / "merged" / "data.yaml",
        "runs_dir":  ML_DIR / "runs" / "security",
        "classes":   ["weak_entry", "fence_gap", "camera_blind_spot"],
    },
}

_SCRATCH_DEFAULTS  = dict(epochs=100, lr0=0.001,  name="v1")
_FINETUNE_DEFAULTS = dict(epochs=30,  lr0=0.0003, name="v2")


def _resolve_weights(model_name: str, weights_arg: str | None) -> str:
    """Return the weights path to use, preferring existing v1 best.pt over scratch."""
    if weights_arg is not None:
        return weights_arg
    v1 = MODELS[model_name]["runs_dir"] / "v1" / "weights" / "best.pt"
    if v1.exists():
        return str(v1)
    return "yolov8n.pt"


def train(model_name: str, weights: str, epochs, lr0, name) -> None:
    cfg = MODELS[model_name]
    data_yaml: Path = cfg["data_yaml"]

    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found.")
        print(f"       Run: python merge_{model_name}.py")
        raise SystemExit(1)

    fine_tune = weights != "yolov8n.pt"
    defaults  = _FINETUNE_DEFAULTS if fine_tune else _SCRATCH_DEFAULTS

    resolved_epochs = epochs if epochs is not None else defaults["epochs"]
    resolved_lr0    = lr0    if lr0    is not None else defaults["lr0"]
    resolved_name   = name   if name   is not None else defaults["name"]

    if fine_tune:
        weights_path = Path(weights)
        if not weights_path.exists():
            print(f"ERROR: Checkpoint not found: {weights_path}")
            raise SystemExit(1)

    device       = 0 if torch.cuda.is_available() else "cpu"
    device_label = f"GPU (cuda:{device})" if isinstance(device, int) else "CPU"
    mode_label   = f"Fine-tune from {weights}" if fine_tune else "Train from scratch (yolov8n.pt)"

    print("=" * 60)
    print(f"  Proptyze — {model_name.capitalize()} Model Training")
    print("=" * 60)
    print(f"  Mode    : {mode_label}")
    print(f"  Classes : {cfg['classes']}")
    print(f"  Dataset : {data_yaml}")
    print(f"  Epochs  : {resolved_epochs}")
    print(f"  lr0     : {resolved_lr0}")
    print(f"  Run name: {resolved_name}")
    print(f"  Device  : {device_label}\n")

    model = YOLO(weights)

    model.train(
        data=str(data_yaml),
        epochs=resolved_epochs,
        imgsz=640,
        batch=8,
        patience=30,
        device=device,
        project=str(cfg["runs_dir"]),
        name=resolved_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=resolved_lr0,
        cos_lr=True,
        plots=True,
        val=True,
        save=True,
        verbose=True,
        workers=0,
    )

    best = cfg["runs_dir"] / resolved_name / "weights" / "best.pt"
    print(f"\nBest weights → {best}")
    print(f"Evaluate with: python evaluate.py --model {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["condition", "security"],
        default="condition",
        help="Which model to train (default: condition)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Starting weights (default: v1/weights/best.pt if it exists, "
            "otherwise yolov8n.pt for a first-time scratch train)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (default: 100 for scratch, 30 for fine-tune)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Initial learning rate (default: 0.001 for scratch, 0.0003 for fine-tune)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name / output folder (default: 'v1' for scratch, 'v2' for fine-tune)",
    )
    args = parser.parse_args()
    weights = _resolve_weights(args.model, args.weights)
    train(args.model, weights, args.epochs, args.lr0, args.name)
