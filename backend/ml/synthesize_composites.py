"""
synthesize_composites.py — Paste small mould patches onto clean background crops
to manufacture walkthrough-distance training examples.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ML_DIR      = Path(__file__).parent
MOULD_TRAIN = ML_DIR / "datasets" / "condition" / "mould" / "train"
BG_SOURCES  = [
    ML_DIR / "datasets" / "condition" / "wall_crack"     / "train",
    ML_DIR / "datasets" / "condition" / "peeling_paint_1" / "train",
    ML_DIR / "datasets" / "condition" / "peeling_paint_2" / "train",
]
OUT_DIR     = ML_DIR / "datasets" / "condition" / "mould_synthetic" / "train"

BG_SIZE     = 640      # output composite size
SCALE_MIN   = 0.03     # mould patch as fraction of BG_SIZE width
SCALE_MAX   = 0.12
PATCH_PAD   = 0.05     # fractional padding around each labelled bbox when cropping
MAX_BG_TRIES = 20      # attempts to find a non-overlapping bg crop


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_boxes(label_path: Path, img_w: int, img_h: int):
    """Return list of (x1,y1,x2,y2) pixel boxes from a YOLO label file."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:])
        x1 = max(0, int((cx - bw / 2) * img_w))
        y1 = max(0, int((cy - bh / 2) * img_h))
        x2 = min(img_w, int((cx + bw / 2) * img_w))
        y2 = min(img_h, int((cy + bh / 2) * img_h))
        boxes.append((x1, y1, x2, y2))
    return boxes


def _rects_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> bool:
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)


# ── patch pool ───────────────────────────────────────────────────────────────

def build_patch_pool(mould_train: Path) -> list:
    """
    Crop every labelled mould bbox (with PATCH_PAD padding) from mould/train.
    Returns a list of BGR numpy arrays.
    """
    img_dir = mould_train / "images"
    lbl_dir = mould_train / "labels"
    patches = []
    img_paths = sorted(img_dir.glob("*.jpg"))
    n = len(img_paths)
    print(f"  Building patch pool from {n} mould/train images …")

    for i, img_path in enumerate(img_paths):
        if i % 500 == 0:
            print(f"    [{i}/{n}]")
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, bw, bh = map(float, parts[1:])
            # expand bbox by padding
            bw_p = min(1.0, bw + PATCH_PAD * 2)
            bh_p = min(1.0, bh + PATCH_PAD * 2)
            x1 = max(0, int((cx - bw_p / 2) * w))
            y1 = max(0, int((cy - bh_p / 2) * h))
            x2 = min(w, int((cx + bw_p / 2) * w))
            y2 = min(h, int((cy + bh_p / 2) * h))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                patches.append(img[y1:y2, x1:x2].copy())

    print(f"  Patch pool: {len(patches)} patches extracted.")
    return patches


# ── background pool ───────────────────────────────────────────────────────────

def build_bg_list(sources: list) -> list:
    """Return list of (image_path, label_path) for all background images."""
    entries = []
    for src in sources:
        img_dir = src / "images"
        lbl_dir = src / "labels"
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            entries.append((img_path, lbl_path))
    print(f"  Background pool: {len(entries)} images across {len(sources)} sources.")
    return entries


def get_bg_crop(img_path: Path, lbl_path: Path) -> np.ndarray | None:
    """
    Load image, randomly crop a BG_SIZE×BG_SIZE region that avoids labelled boxes.
    Falls back to a random crop if no clean region found.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    boxes = _parse_boxes(lbl_path, w, h)

    crop_w = min(w, BG_SIZE)
    crop_h = min(h, BG_SIZE)

    for _ in range(MAX_BG_TRIES):
        x_off = random.randint(0, max(0, w - crop_w))
        y_off = random.randint(0, max(0, h - crop_h))
        cx1, cy1 = x_off, y_off
        cx2, cy2 = x_off + crop_w, y_off + crop_h

        if not any(_rects_overlap(cx1, cy1, cx2, cy2, *b) for b in boxes):
            crop = img[y_off:y_off + crop_h, x_off:x_off + crop_w]
            if crop_w < BG_SIZE or crop_h < BG_SIZE:
                crop = cv2.resize(crop, (BG_SIZE, BG_SIZE), interpolation=cv2.INTER_LINEAR)
            return crop

    # fallback – just take a random crop regardless of overlap
    x_off = random.randint(0, max(0, w - crop_w))
    y_off = random.randint(0, max(0, h - crop_h))
    crop  = img[y_off:y_off + crop_h, x_off:x_off + crop_w]
    if crop_w < BG_SIZE or crop_h < BG_SIZE:
        crop = cv2.resize(crop, (BG_SIZE, BG_SIZE), interpolation=cv2.INTER_LINEAR)
    return crop


# ── composite ─────────────────────────────────────────────────────────────────

def make_composite(bg: np.ndarray, patch: np.ndarray):
    """
    Paste `patch` onto `bg` at a random position and scale.
    Returns (composite_bgr, yolo_label_str, actual_scale_frac) or (None, None, 0).
    """
    bg_h, bg_w = bg.shape[:2]
    scale = random.uniform(SCALE_MIN, SCALE_MAX)
    target_w = max(8, int(scale * bg_w))

    ph, pw = patch.shape[:2]
    target_h = max(8, int(target_w * ph / pw))

    if target_w >= bg_w or target_h >= bg_h:
        return None, None, 0.0

    # resize patch
    scaled = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # brightness/contrast jitter to roughly match background lighting
    alpha_j = random.uniform(0.75, 1.25)
    beta_j  = random.randint(-25, 25)
    scaled  = np.clip(scaled.astype(np.float32) * alpha_j + beta_j, 0, 255).astype(np.uint8)

    # random paste position
    x1 = random.randint(0, bg_w - target_w)
    y1 = random.randint(0, bg_h - target_h)
    x2, y2 = x1 + target_w, y1 + target_h

    # feathered alpha mask — blur a solid mask so edges fade out
    mask    = np.ones((target_h, target_w), dtype=np.float32)
    feather = max(1, min(target_h, target_w) // 6)
    ks      = feather * 2 + 1
    mask    = cv2.GaussianBlur(mask, (ks, ks), feather / 3.0)
    alpha   = mask[:, :, np.newaxis]

    comp    = bg.copy()
    roi     = comp[y1:y2, x1:x2].astype(np.float32)
    blended = alpha * scaled.astype(np.float32) + (1 - alpha) * roi
    comp[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    # YOLO label: class 0 = mould (sub-dataset local ID, matches mould/ source format)
    cx_n = (x1 + x2) / 2 / bg_w
    cy_n = (y1 + y2) / 2 / bg_h
    w_n  = target_w / bg_w
    h_n  = target_h / bg_h
    label = f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"

    return comp, label, scale


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000,
                        help="Number of composites to generate (default: 1000)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_img = OUT_DIR / "images"
    out_lbl = OUT_DIR / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Proptyze — Synthetic Mould Composites")
    print("=" * 60)
    print(f"  Target  : {args.target} composites")
    print(f"  Seed    : {args.seed}")
    print(f"  Output  : {OUT_DIR}\n")

    patches = build_patch_pool(MOULD_TRAIN)
    if not patches:
        raise SystemExit("ERROR: no mould patches extracted — check mould/train/labels")

    bg_list = build_bg_list(BG_SOURCES)
    if not bg_list:
        raise SystemExit("ERROR: no background images found")

    print(f"\n  Generating composites …")
    scales_pct   = []
    generated    = 0
    attempts     = 0
    max_attempts = args.target * 5

    while generated < args.target and attempts < max_attempts:
        attempts += 1

        patch   = random.choice(patches)
        bg_path, lbl_path = random.choice(bg_list)
        bg      = get_bg_crop(bg_path, lbl_path)
        if bg is None:
            continue

        comp, label, scale = make_composite(bg, patch)
        if comp is None:
            continue

        idx      = generated
        img_name = f"synth_mould_{idx:05d}.jpg"
        lbl_name = f"synth_mould_{idx:05d}.txt"

        cv2.imwrite(str(out_img / img_name), comp, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_lbl / lbl_name).write_text(label)

        scales_pct.append(scale * 100)
        generated += 1

        if generated % 100 == 0:
            print(f"    [{generated}/{args.target}]  attempts so far: {attempts}")

    print(f"\n{'=' * 60}")
    print(f"  Done — {generated} composites saved to {OUT_DIR}")
    print(f"{'=' * 60}")

    if scales_pct:
        import statistics
        pct_bins = [0] * 5   # <4, 4-6, 6-8, 8-10, 10-12+
        for s in scales_pct:
            if   s < 4:  pct_bins[0] += 1
            elif s < 6:  pct_bins[1] += 1
            elif s < 8:  pct_bins[2] += 1
            elif s < 10: pct_bins[3] += 1
            else:        pct_bins[4] += 1

        print(f"\n  Patch width as % of image width (confirms walkthrough scale):")
        labels = ["<4%", "4–6%", "6–8%", "8–10%", "10–12%+"]
        for lbl, cnt in zip(labels, pct_bins):
            bar = "#" * (cnt * 30 // max(pct_bins or [1]))
            print(f"    {lbl:>8}  {cnt:>5}  {bar}")
        print(f"\n  Min: {min(scales_pct):.1f}%   Max: {max(scales_pct):.1f}%   "
              f"Median: {statistics.median(scales_pct):.1f}%")
        print(f"\n  Next step: inspect a few images in {out_img}")
        print(f"  Then add mould_synthetic to DATASET_CLASS_MAPS in merge_condition.py")
        print(f"  and re-run: python merge_condition.py")


if __name__ == "__main__":
    main()
