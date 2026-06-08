"""
evaluate.py — Evaluate a trained model and print per-class metrics.

Run from: backend/ml/
    python evaluate.py --model condition
    python evaluate.py --model security
"""

import argparse
from pathlib import Path

try:
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


def find_weights(runs_dir: Path) -> Path:
    default = runs_dir / "v1" / "weights" / "best.pt"
    if default.exists():
        return default
    candidates = sorted(runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else default


def evaluate(model_name: str) -> None:
    cfg       = MODELS[model_name]
    data_yaml = cfg["data_yaml"]
    weights   = find_weights(cfg["runs_dir"])
    classes   = cfg["classes"]

    if not weights.exists():
        print(f"ERROR: No weights found at {weights}")
        print(f"       Run: python train.py --model {model_name}")
        return

    if not data_yaml.exists():
        print(f"ERROR: No data.yaml at {data_yaml}")
        print(f"       Run: python merge_{model_name}.py")
        return

    print("=" * 60)
    print(f"  Proptyze — {model_name.capitalize()} Model Evaluation")
    print("=" * 60)
    print(f"  Weights : {weights}")
    print(f"  Dataset : {data_yaml}\n")

    model   = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), plots=True, save_json=True, verbose=False)

    box       = metrics.box
    names     = metrics.names
    evaluated = list(metrics.box.ap_class_index) if hasattr(box, "ap_class_index") else []

    print("\n" + "=" * 60)
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("  " + "-" * 58)

    for j, cls_idx in enumerate(evaluated):
        cls_name = names.get(int(cls_idx), f"class_{cls_idx}")
        p    = float(box.p[j])    if box.p    is not None else float("nan")
        r    = float(box.r[j])    if box.r    is not None else float("nan")
        ap50 = float(box.ap50[j]) if box.ap50 is not None else float("nan")
        ap   = float(box.ap[j])   if box.ap   is not None else float("nan")
        print(f"  {cls_name:<20} {p:>10.3f} {r:>8.3f} {ap50:>8.3f} {ap:>10.3f}")

    evaluated_names = {names.get(int(i)) for i in evaluated}
    for cls_name in classes:
        if cls_name not in evaluated_names:
            print(f"  {cls_name:<20} {'(no val data)':>39}")

    print("  " + "-" * 58)
    print(f"  {'ALL (mean)':<20} {box.mp:>10.3f} {box.mr:>8.3f} {box.map50:>8.3f} {box.map:>10.3f}")
    print("=" * 60)

    print("\nSanity check:")
    if box.map50 >= 0.7:
        print(f"  mAP50 {box.map50:.3f} — looks good, ready to integrate.")
    elif box.map50 >= 0.5:
        print(f"  mAP50 {box.map50:.3f} — reasonable start; more data or epochs will help.")
    else:
        print(f"  mAP50 {box.map50:.3f} — low; check class mappings and dataset quality.")

    print(f"\nResults and plots → {weights.parent.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["condition", "security"],
        default="condition",
        help="Which model to evaluate (default: condition)",
    )
    args = parser.parse_args()
    evaluate(args.model)
