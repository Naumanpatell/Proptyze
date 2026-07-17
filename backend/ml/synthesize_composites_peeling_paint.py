"""
synthesize_composites_peeling_paint.py

Takes peeling paint patches (cropped from the peeling_paint_1 training
set) and pastes them onto clean backgrounds from wall_crack, damp, and
mould. This creates extra training examples for the peeling paint
class specifically.
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ML_DIR = Path(__file__).parent

# source dataset to pull peeling paint patches from
PEELING_SOURCES = [
    ML_DIR / "datasets" / "condition" / "peeling_paint_1" / "train",
]

PEELING_CLASS_IDXS = None

# folders we pull clean backgrounds from (images that don't have peeling paint)
BG_SOURCES = [
    ML_DIR / "datasets" / "condition" / "wall_crack" / "train",
    ML_DIR / "datasets" / "condition" / "damp" / "train",
    ML_DIR / "datasets" / "condition" / "mould" / "train",
]
OUT_DIR = ML_DIR / "datasets" / "condition" / "peeling_paint_synthetic" / "train"

BG_SIZE = 640
SCALE_MIN = 0.08   # peeling paint patches tend to be mid-to-large, similar to damp
SCALE_MAX = 0.32
PATCH_PAD = 0.05
MAX_BG_TRIES = 20


def _parse_boxes(label_path: Path, img_w: int, img_h: int):
    """Read a YOLO label file and convert to pixel coordinates."""
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


def build_patch_pool(sources: list) -> list:
    """Go through the peeling paint training images and cut out the patches."""
    patches = []
    for src in sources:
        img_dir = src / "images"
        lbl_dir = src / "labels"
        img_paths = sorted(img_dir.glob("*.jpg"))
        n = len(img_paths)
        print(f"Reading {n} images from {src}...")

        for i, img_path in enumerate(img_paths):
            if i % 500 == 0:
                print(f"  {i}/{n} done")
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
                cls_idx = int(parts[0])
                if PEELING_CLASS_IDXS is not None and cls_idx not in PEELING_CLASS_IDXS:
                    continue
                cx, cy, bw, bh = map(float, parts[1:])
                bw_p = min(1.0, bw + PATCH_PAD * 2)
                bh_p = min(1.0, bh + PATCH_PAD * 2)
                x1 = max(0, int((cx - bw_p / 2) * w))
                y1 = max(0, int((cy - bh_p / 2) * h))
                x2 = min(w, int((cx + bw_p / 2) * w))
                y2 = min(h, int((cy + bh_p / 2) * h))
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    crop = img[y1:y2, x1:x2]
                    # cap stored size to limit RAM usage
                    if crop.shape[1] > 250:
                        scale_f = 250 / crop.shape[1]
                        crop = cv2.resize(crop, (250, max(8, int(crop.shape[0] * scale_f))), interpolation=cv2.INTER_AREA)
                    patches.append(crop.copy())

    print(f"Got {len(patches)} peeling paint patches total.")
    return patches


def build_bg_list(sources: list) -> list:
    """Collect every image+label pair from the background source folders."""
    entries = []
    for src in sources:
        img_dir = src / "images"
        lbl_dir = src / "labels"
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            entries.append((img_path, lbl_path))
    print(f"Found {len(entries)} background images across {len(sources)} folders.")
    return entries


def get_bg_crop(img_path: Path, lbl_path: Path) -> np.ndarray | None:
    """
    Pick a random crop from a background image, trying to avoid any area
    that already has a labeled defect in it.
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

    # couldn't find a clean spot, just use a random crop anyway
    x_off = random.randint(0, max(0, w - crop_w))
    y_off = random.randint(0, max(0, h - crop_h))
    crop = img[y_off:y_off + crop_h, x_off:x_off + crop_w]
    if crop_w < BG_SIZE or crop_h < BG_SIZE:
        crop = cv2.resize(crop, (BG_SIZE, BG_SIZE), interpolation=cv2.INTER_LINEAR)
    return crop


def make_composite(bg: np.ndarray, patch: np.ndarray):
    """Paste one patch onto one background at a random size/position/brightness."""
    bg_h, bg_w = bg.shape[:2]
    scale = random.uniform(SCALE_MIN, SCALE_MAX)
    target_w = max(8, int(scale * bg_w))

    ph, pw = patch.shape[:2]
    target_h = max(8, int(target_w * ph / pw))

    if target_w >= bg_w or target_h >= bg_h:
        return None, None, 0.0

    scaled = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_AREA)

    alpha_j = random.uniform(0.75, 1.25)
    beta_j = random.randint(-25, 25)
    scaled = np.clip(scaled.astype(np.float32) * alpha_j + beta_j, 0, 255).astype(np.uint8)

    x1 = random.randint(0, bg_w - target_w)
    y1 = random.randint(0, bg_h - target_h)
    x2, y2 = x1 + target_w, y1 + target_h

    mask = np.ones((target_h, target_w), dtype=np.float32)
    feather = max(1, min(target_h, target_w) // 6)
    ks = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (ks, ks), feather / 3.0)
    alpha = mask[:, :, np.newaxis]

    comp = bg.copy()
    roi = comp[y1:y2, x1:x2].astype(np.float32)
    blended = alpha * scaled.astype(np.float32) + (1 - alpha) * roi
    comp[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    # class 0 = peeling_paint (local sub-dataset ID)
    cx_n = (x1 + x2) / 2 / bg_w
    cy_n = (y1 + y2) / 2 / bg_h
    w_n = target_w / bg_w
    h_n = target_h / bg_h
    label = f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"

    return comp, label, scale


def split_train_valid(out_dir: Path, train_ratio: float = 0.8):
    """Move 20% of the generated images into a valid/ folder."""
    train_images = out_dir / "images"
    train_labels = out_dir / "labels"

    valid_images = out_dir.parent / "valid" / "images"
    valid_labels = out_dir.parent / "valid" / "labels"

    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)

    all_images = sorted(train_images.glob("*.jpg"))
    random.shuffle(all_images)

    split_idx = int(len(all_images) * train_ratio)

    for img in all_images[split_idx:]:
        label = train_labels / (img.stem + ".txt")
        shutil.move(str(img), str(valid_images / img.name))
        if label.exists():
            shutil.move(str(label), str(valid_labels / label.name))

    train_count = len(list(train_images.glob("*.jpg")))
    valid_count = len(list(valid_images.glob("*.jpg")))

    print(f"Split done - train: {train_count} images, valid: {valid_count} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000,
                        help="Number of composites to generate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_img = OUT_DIR / "images"
    out_lbl = OUT_DIR / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.target} synthetic peeling paint composites (seed={args.seed})")
    print(f"Output folder: {OUT_DIR}")

    patches = build_patch_pool(PEELING_SOURCES)
    if not patches:
        raise SystemExit("No peeling paint patches found - check peeling_paint_1/train/labels folder")

    bg_list = build_bg_list(BG_SOURCES)
    if not bg_list:
        raise SystemExit("No background images found")

    print("Starting generation...")
    scales_pct = []
    generated = 0
    attempts = 0
    max_attempts = args.target * 10

    while generated < args.target and attempts < max_attempts:
        attempts += 1

        patch = random.choice(patches)
        bg_path, lbl_path = random.choice(bg_list)
        bg = get_bg_crop(bg_path, lbl_path)
        if bg is None:
            continue

        comp, label, scale = make_composite(bg, patch)
        if comp is None:
            continue

        img_name = f"synth_peel_{generated:05d}.jpg"
        lbl_name = f"synth_peel_{generated:05d}.txt"

        cv2.imwrite(str(out_img / img_name), comp, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_lbl / lbl_name).write_text(label)

        scales_pct.append(scale * 100)
        generated += 1

        if generated % 100 == 0:
            print(f"  {generated}/{args.target} generated ({attempts} attempts so far)")

    print(f"Done. {generated} composites saved to {OUT_DIR}")

    if scales_pct:
        import statistics
        print(f"Patch size range: {min(scales_pct):.1f}% - {max(scales_pct):.1f}% of image width "
              f"(median {statistics.median(scales_pct):.1f}%)")

    split_train_valid(OUT_DIR)
   