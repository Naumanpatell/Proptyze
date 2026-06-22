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


def train(model_name: str) -> None:
    cfg = MODELS[model_name]
    data_yaml: Path = cfg["data_yaml"]

    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found.")
        print(f"       Run: python merge_{model_name}.py")
        raise SystemExit(1)

    device       = 0 if torch.cuda.is_available() else "cpu"
    device_label = f"GPU (cuda:{device})" if isinstance(device, int) else "CPU"

    print("=" * 60)
    print(f"  Proptyze — {model_name.capitalize()} Model Training")
    print("=" * 60)
    print(f"  Classes : {cfg['classes']}")
    print(f"  Dataset : {data_yaml}")
    print(f"  Device  : {device_label}\n")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=8,
        patience=30,
        device=device,
        project=str(cfg["runs_dir"]),
        name="v1",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        plots=True,
        val=True,
        save=True,
        verbose=True,
        workers=0,
    )

    best = cfg["runs_dir"] / "v1" / "weights" / "best.pt"
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
    args = parser.parse_args()
4    train(args.model)
