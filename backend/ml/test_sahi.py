import argparse
from collections import defaultdict
from pathlib import Path

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ERROR: pip install ultralytics")

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError:
    raise SystemExit("ERROR: pip install sahi")

ML_DIR        = Path(__file__).parent
DEFAULT_WGTS  = ML_DIR / "runs" / "condition" / "v1" / "weights" / "best.pt"
ALL_CLASSES   = ["wall_crack", "damp", "mould", "peeling_paint", "broken_fixture"]


def _device_str():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def run_standard(weights: Path, image: Path, conf: float):
    """Direct YOLO inference on the full image. Returns list of (name, conf, xyxy)."""
    model   = YOLO(str(weights))
    results = model.predict(source=str(image), conf=conf, verbose=False)
    dets    = []
    for box in results[0].boxes:
        cls_name = results[0].names[int(box.cls)]
        dets.append((cls_name, float(box.conf), [round(v) for v in box.xyxy[0].tolist()]))
    return dets


def run_sahi(weights: Path, image: Path, conf: float,
             slice_h: int, slice_w: int, overlap_h: float, overlap_w: float):
    """SAHI sliced inference. Returns list of (name, conf, xyxy)."""
    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=conf,
        device=_device_str(),
    )
    result = get_sliced_prediction(
        image=str(image),
        detection_model=det_model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap_h,
        overlap_width_ratio=overlap_w,
        verbose=0,
    )
    dets = []
    for pred in result.object_prediction_list:
        xyxy = [round(v) for v in pred.bbox.to_xyxy()]
        dets.append((pred.category.name, round(pred.score.value, 4), xyxy))
    return dets


def print_dets(label: str, dets: list):
    print(f"\n  {'Class':<20} {'Conf':>6}  BBox [x1,y1,x2,y2]")
    print("  " + "-" * 60)
    if not dets:
        print("  (no detections)")
        return
    for name, conf, bbox in sorted(dets, key=lambda d: -d[1]):
        print(f"  {name:<20} {conf:>6.3f}  {bbox}")


def print_summary(std_dets: list, sahi_dets: list):
    def tally(dets):
        counts = defaultdict(int)
        confs  = defaultdict(list)
        for name, conf, _ in dets:
            counts[name] += 1
            confs[name].append(conf)
        return counts, confs

    sc, sf_c = tally(std_dets)
    hc, hf_c = tally(sahi_dets)

    print("\n" + "=" * 68)
    print("  SUMMARY: Standard  vs  SAHI-sliced")
    print("=" * 68)
    print(f"  {'Class':<20} {'Std cnt':>7} {'Std conf':>9}   {'SAHI cnt':>8} {'SAHI conf':>10}  {'Δ':>4}")
    print("  " + "-" * 66)

    total_s = total_h = 0
    for cls in ALL_CLASSES:
        s_n   = sc.get(cls, 0)
        h_n   = hc.get(cls, 0)
        s_avg = f"{sum(sf_c[cls])/len(sf_c[cls]):.3f}" if sf_c.get(cls) else "  —  "
        h_avg = f"{sum(hf_c[cls])/len(hf_c[cls]):.3f}" if hf_c.get(cls) else "  —  "
        delta = h_n - s_n
        d_str = f"{delta:+d}" if delta != 0 else " 0"
        print(f"  {cls:<20} {s_n:>7}  {s_avg:>8}   {h_n:>8}  {h_avg:>9}  {d_str:>4}")
        total_s += s_n
        total_h += h_n

    print("  " + "-" * 66)
    print(f"  {'TOTAL':<20} {total_s:>7}  {'':>8}   {total_h:>8}  {'':>9}  {total_h-total_s:>+4}")
    print("=" * 68)

    mould_gain = hc.get("mould", 0) - sc.get("mould", 0)
    damp_gain  = hc.get("damp",  0) - sc.get("damp",  0)
    print(f"\n  mould delta: {mould_gain:+d}   damp delta: {damp_gain:+d}")
    if mould_gain > 0 or damp_gain > 0:
        print("  SAHI slicing found additional mould/damp detections — slicing likely helps.")
    elif mould_gain == 0 and damp_gain == 0:
        print("  No difference in mould/damp count — slicing did not help for this image.")
    else:
        print("  SAHI found fewer mould/damp — NMS may be over-suppressing; try lower iou_threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",         required=True, help="Path to test image")
    parser.add_argument("--weights",       default=str(DEFAULT_WGTS),
                        help=f"Weights path (default: {DEFAULT_WGTS})")
    parser.add_argument("--conf",          type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--slice-height",  type=int,   default=320,
                        help="SAHI slice height (default: 320)")
    parser.add_argument("--slice-width",   type=int,   default=320,
                        help="SAHI slice width (default: 320)")
    parser.add_argument("--overlap-h",     type=float, default=0.2,
                        help="SAHI overlap height ratio (default: 0.2)")
    parser.add_argument("--overlap-w",     type=float, default=0.2,
                        help="SAHI overlap width ratio (default: 0.2)")
    args = parser.parse_args()

    image   = Path(args.image)
    weights = Path(args.weights)

    if not image.exists():
        raise SystemExit(f"ERROR: image not found: {image}")
    if not weights.exists():
        raise SystemExit(f"ERROR: weights not found: {weights}\n"
                         "       Run: python train.py --model condition")

    print("=" * 68)
    print(f"  Image   : {image}")
    print(f"  Weights : {weights}")
    print(f"  Conf    : {args.conf}   Slices: {args.slice_height}×{args.slice_width}  "
          f"Overlap: {args.overlap_h}/{args.overlap_w}")
    print("=" * 68)

    print("\n[STANDARD — whole image]")
    std_dets = run_standard(weights, image, args.conf)
    print_dets("STANDARD", std_dets)

    print(f"\n[SAHI SLICED — {args.slice_height}×{args.slice_width} tiles, "
          f"overlap {args.overlap_h}/{args.overlap_w}]")
    sahi_dets = run_sahi(weights, image, args.conf,
                         args.slice_height, args.slice_width,
                         args.overlap_h, args.overlap_w)
    print_dets("SAHI", sahi_dets)

    print_summary(std_dets, sahi_dets)


if __name__ == "__main__":
    main()
