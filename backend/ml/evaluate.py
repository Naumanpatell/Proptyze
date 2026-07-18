"""
evaluate.py — Evaluate a trained model and print per-class metrics.
Automatically finds the latest version and shows current status.
"""

import argparse
import math
from pathlib import Path
from datetime import datetime

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
    "condition_peeling": {
        "data_yaml": ML_DIR / "datasets" / "condition" / "merged_peeling" / "data.yaml",
        "runs_dir":  ML_DIR / "runs" / "condition_peeling",
        "classes":   ["wall_crack", "damp", "mould", "peeling_paint", "broken_fixture"],
    },
    "security": {
        "data_yaml": ML_DIR / "datasets" / "security" / "merged" / "data.yaml",
        "runs_dir":  ML_DIR / "runs" / "security",
        "classes":   ["weak_entry", "fence_gap", "camera_blind_spot"],
    },
}

# Which merge script to point people to when data.yaml is missing
_MERGE_SCRIPT = {
    "condition":         "merge_condition.py",
    "condition_peeling": "merge_condition_peeling.py",
    "security":          "merge_security.py",
}


def find_latest_version(runs_dir: Path) -> tuple:
    """Find the latest trained version by modification time"""
    candidates = list(runs_dir.glob("v*/weights/best.pt"))
    if not candidates:
        return None, None
    
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    version = latest.parent.parent.name  # e.g., "v2"
    mtime = datetime.fromtimestamp(latest.stat().st_mtime)
    return latest, mtime


def run_val(weights: Path, data_yaml: Path) -> tuple:
    """Run validation; return (per_class_dict, overall_dict)"""
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), plots=False, save_json=False, verbose=False, workers=0)

    box = metrics.box
    names = metrics.names
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


def print_model_status(model_name: str, weights: Path, mtime: datetime, per_class: dict, overall: dict, classes: list):
    """Print a single model's status"""
    print(f"\n{'='*90}")
    print(f"  MODEL: {model_name.upper()}")
    print(f"  WEIGHTS: {weights}")
    print(f"  TRAINED: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*90}")
    
    print(f"\n  {'Class':<20} {'Precision':>12} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>12}")
    print(f"  {'-'*90}")
    
    for cls_name in classes:
        if cls_name in per_class:
            d = per_class[cls_name]
            p_str = f"{d['p']:.3f}" if not math.isnan(d['p']) else "—"
            r_str = f"{d['r']:.3f}" if not math.isnan(d['r']) else "—"
            ap50_str = f"{d['ap50']:.3f}" if not math.isnan(d['ap50']) else "—"
            ap_str = f"{d['ap']:.3f}" if not math.isnan(d['ap']) else "—"
            print(f"  {cls_name:<20} {p_str:>12} {r_str:>10} {ap50_str:>10} {ap_str:>12}")
        else:
            print(f"  {cls_name:<20} {'(no validation data)':>44}")
    
    print(f"  {'-'*90}")
    print(f"  {'OVERALL (mean)':<20} {overall['mp']:>12.3f} {overall['mr']:>10.3f} {overall['map50']:>10.3f} {overall['map']:>12.3f}")
    print(f"{'='*90}\n")


def _print_comparison(classes: list, pc1: dict, ov1: dict, pc2: dict, ov2: dict,
                      label1: str, label2: str) -> None:
    """Side-by-side comparison table with ΔRecall column."""
    CW    = 20   # class name width
    VW    = 7    # per-metric column width
    BLOCK = VW * 4 + 3
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
    print(f"  COMPARISON: {label1}  vs  {label2}")
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

    # Focus on problem classes
    focus = [c for c in ("mould", "damp", "peeling_paint") if c in classes]
    if focus:
        print("\nProblem class recall (key focus areas):")
        for cls_name in focus:
            d1 = pc1.get(cls_name)
            d2 = pc2.get(cls_name)
            if d1 and d2:
                r1, r2 = d1["r"], d2["r"]
                arrow  = "▲" if r2 > r1 else ("▼" if r2 < r1 else "—")
                print(f"  {cls_name:<20} {r1:.3f} → {r2:.3f}  ({arrow} {r2 - r1:+.3f})")
            else:
                print(f"  {cls_name:<20} (no val data for one or both versions)")


def evaluate(model_name: str) -> None:
    cfg       = MODELS[model_name]
    data_yaml = cfg["data_yaml"]
    runs_dir  = cfg["runs_dir"]
    classes   = cfg["classes"]

    if not data_yaml.exists():
        print(f"ERROR: No data.yaml at {data_yaml}")
        print(f"       Run: python {_MERGE_SCRIPT.get(model_name, 'merge_' + model_name + '.py')}")
        return

    print(f"\n{'='*90}")
    print(f"  PROPTYZE {model_name.upper()} MODEL — CURRENT STATUS")
    print(f"{'='*90}")

    # Find the LATEST version
    weights, mtime = find_latest_version(runs_dir)
    if not weights:
        print(f"ERROR: No weights found in {runs_dir}")
        print(f"       Run: python train.py --model {model_name}")
        return
    
    version = weights.parent.parent.name
    print(f"  Evaluating latest version: {version}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Trained: {mtime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    per_class, overall = run_val(weights, data_yaml)
    print_model_status(model_name, weights, mtime, per_class, overall, classes)
    
    # Special check for condition_peeling: also evaluate on FULL condition dataset
    if model_name == "condition_peeling":
        print(f"\n  REGRESSION CHECK: Evaluating on FULL condition dataset...")
        full_dataset = ML_DIR / "datasets" / "condition" / "merged" / "data.yaml"
        if full_dataset.exists():
            per_class_full, overall_full = run_val(weights, full_dataset)
            print_model_status(f"{model_name} (full dataset)", weights, mtime, per_class_full, overall_full, classes)
            
            # Quick comparison
            print(f"\n  {'='*90}")
            print(f"  COMPARISON:")
            print(f"  {'='*90}")
            print(f"  On peeling-focused dataset:  mAP50 = {overall['map50']:.3f}")
            print(f"  On full condition dataset:   mAP50 = {overall_full['map50']:.3f}")
            
            if overall_full['map50'] >= 0.55:
                print(f"  NO REGRESSION! Safe to use in production\n")
            else:
                print(f"  Performance dropped on full dataset - may need retraining\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["condition", "condition_peeling", "security"],
        default="condition",
        help="Which model to evaluate (default: condition)",
    )
    args = parser.parse_args()
    evaluate(args.model)