"""
evaluate.py — Evaluate a trained model and print per-class metrics.

When both v1 and v2 weights exist, automatically prints a side-by-side
comparison table with a ΔRecall column.

Run from: backend/ml/
    python evaluate.py --model condition
    python evaluate.py --model security
"""

import argparse
import math
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


def _run_val(weights: Path, data_yaml: Path) -> tuple:
    """Run validation; return (per_class_dict, overall_dict)."""
    model   = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), plots=True, save_json=True, verbose=False, workers=0)

    box       = metrics.box
    names     = metrics.names
    evaluated = list(box.ap_class_index) if hasattr(box, "ap_class_index") else []

    per_class = {}
    for j, cls_idx in enumerate(evaluated):
        cls_name = names.get(int(cls_idx), f"class_{cls_idx}")
        per_class[cls_name] = {
            "p":    float(box.p[j])    if box.p    is not None else math.nan,
            "r":    float(box.r[j])    if box.r    is not None else math.nan,
            "ap50": float(box.ap50[j]) if box.ap50 is not None else math.nan,
            "ap":   float(box.ap[j])   if box.ap   is not None else math.nan,
        }
    overall = {
        "mp":    float(box.mp),
        "mr":    float(box.mr),
        "map50": float(box.map50),
        "map":   float(box.map),
    }
    return per_class, overall


def _print_single(weights: Path, classes: list, per_class: dict, overall: dict, model_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("  " + "-" * 58)

    for cls_name in classes:
        if cls_name in per_class:
            d = per_class[cls_name]
            print(f"  {cls_name:<20} {d['p']:>10.3f} {d['r']:>8.3f} {d['ap50']:>8.3f} {d['ap']:>10.3f}")
        else:
            print(f"  {cls_name:<20} {'(no val data)':>39}")

    print("  " + "-" * 58)
    print(f"  {'ALL (mean)':<20} {overall['mp']:>10.3f} {overall['mr']:>8.3f} {overall['map50']:>8.3f} {overall['map']:>10.3f}")
    print("=" * 60)

    print("\nSanity check:")
    m = overall["map50"]
    if m >= 0.7:
        print(f"  mAP50 {m:.3f} — looks good, ready to integrate.")
    elif m >= 0.5:
        print(f"  mAP50 {m:.3f} — reasonable start; more data or epochs will help.")
    else:
        print(f"  mAP50 {m:.3f} — low; check class mappings and dataset quality.")

    print(f"\nResults and plots → {weights.parent.parent}")


def _print_comparison(classes: list, pc1: dict, ov1: dict, pc2: dict, ov2: dict,
                      label1: str, label2: str) -> None:
    """Side-by-side comparison table with ΔRecall column."""
    CW    = 20   # class name width
    VW    = 7    # per-metric column width
    # one block of 4 value cols: "p r ap50 ap" = VW*4 + 3 spaces = 31
    BLOCK = VW * 4 + 3
    # full row: 2 indent + CW + 2 + BLOCK + 2 + BLOCK + 2 + 8 ΔRecall = 98
    SEP   = 98

    def fv(d, key):
        v = d.get(key, math.nan) if d else math.nan
        return f"{v:{VW}.3f}" if not math.isnan(v) else f"{'—':>{VW}}"

    def vals(d):
        return f"{fv(d,'p')} {fv(d,'r')} {fv(d,'ap50')} {fv(d,'ap')}"

    def delta_r(d1, d2):
        a = d1.get("r", math.nan) if d1 else math.nan
        b = d2.get("r", math.nan) if d2 else math.nan
        if math.isnan(a) or math.isnan(b):
            return f"{'—':>8}"
        return f"{b - a:>+8.3f}"

    print("\n" + "=" * SEP)
    print(f"  Proptyze — Comparison: {label1}  vs  {label2}")
    print("=" * SEP)

    blk1 = f"{'— ' + label1 + ' —':^{BLOCK}}"
    blk2 = f"{'— ' + label2 + ' —':^{BLOCK}}"
    print(f"  {'':>{CW}}  {blk1}  {blk2}  {'':>8}")

    hdr = f"{'P':>{VW}} {'R':>{VW}} {'mAP50':>{VW}} {'mAP':>{VW}}"
    print(f"  {'Class':<{CW}}  {hdr}  {hdr}  {'ΔRecall':>8}")
    print("  " + "-" * (SEP - 2))

    for cls_name in classes:
        d1 = pc1.get(cls_name)
        d2 = pc2.get(cls_name)
        print(f"  {cls_name:<{CW}}  {vals(d1)}  {vals(d2)}  {delta_r(d1, d2)}")

    print("  " + "-" * (SEP - 2))
    o1 = {"p": ov1["mp"], "r": ov1["mr"], "ap50": ov1["map50"], "ap": ov1["map"]}
    o2 = {"p": ov2["mp"], "r": ov2["mr"], "ap50": ov2["map50"], "ap": ov2["map"]}
    dr_all = f"{ov2['mr'] - ov1['mr']:>+8.3f}"
    print(f"  {'ALL (mean)':<{CW}}  {vals(o1)}  {vals(o2)}  {dr_all}")
    print("=" * SEP)

    # Mould/damp spotlight (only printed when those classes exist in the model)
    focus = [c for c in ("mould", "damp") if c in classes]
    if focus:
        print("\nMould / damp recall:")
        for cls_name in focus:
            d1 = pc1.get(cls_name)
            d2 = pc2.get(cls_name)
            if d1 and d2:
                r1, r2 = d1["r"], d2["r"]
                arrow  = "▲" if r2 > r1 else ("▼" if r2 < r1 else "—")
                print(f"  {cls_name:<20} {r1:.3f} → {r2:.3f}  ({arrow} {r2 - r1:+.3f})")
            else:
                print(f"  {cls_name:<20} (no val data for one or both versions)")

    # Regression check for the other classes
    other = [c for c in classes if c not in set(focus)]
    if other:
        print("\nRegression check (other classes):")
        for cls_name in other:
            d1 = pc1.get(cls_name)
            d2 = pc2.get(cls_name)
            if d1 and d2:
                dr = d2["r"]    - d1["r"]
                da = d2["ap50"] - d1["ap50"]
                tag = "REGRESSED" if (dr < -0.05 or da < -0.05) else "OK        "
                print(f"  {cls_name:<20} {tag}  ΔRecall={dr:+.3f}  ΔmAP50={da:+.3f}")
            else:
                print(f"  {cls_name:<20} (no val data for one or both versions)")


def evaluate(model_name: str, v1_name: str = "v1", v2_name: str = "v2") -> None:
    cfg       = MODELS[model_name]
    data_yaml = cfg["data_yaml"]
    runs_dir  = cfg["runs_dir"]
    classes   = cfg["classes"]

    if not data_yaml.exists():
        print(f"ERROR: No data.yaml at {data_yaml}")
        print(f"       Run: python merge_{model_name}.py")
        return

    print("=" * 60)
    print(f"  Proptyze — {model_name.capitalize()} Model Evaluation")
    print("=" * 60)

    v1 = runs_dir / v1_name / "weights" / "best.pt"
    v2 = runs_dir / v2_name / "weights" / "best.pt"

    if v1.exists() and v2.exists():
        print(f"  Found {v1_name} and {v2_name} — running side-by-side comparison.\n")
        print(f"  Evaluating {v1} …")
        pc1, ov1 = _run_val(v1, data_yaml)
        print(f"  Evaluating {v2} …")
        pc2, ov2 = _run_val(v2, data_yaml)
        _print_comparison(classes, pc1, ov1, pc2, ov2, v1_name, v2_name)
    else:
        weights = v1 if v1.exists() else find_weights(runs_dir)
        if not weights.exists():
            print(f"ERROR: No weights found at {weights}")
            print(f"       Run: python train.py --model {model_name}")
            return
        print(f"  Weights : {weights}")
        print(f"  Dataset : {data_yaml}\n")
        per_class, overall = _run_val(weights, data_yaml)
        _print_single(weights, classes, per_class, overall, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["condition", "security"], default="condition")
    parser.add_argument("--v1-name", default="v1", help="First version to compare (default: v1)")
    parser.add_argument("--v2-name", default="v2", help="Second version to compare (default: v2)")
    args = parser.parse_args()
    evaluate(args.model, args.v1_name, args.v2_name)
