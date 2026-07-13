"""
synthesize_composites_camera.py — Paste camera patches onto wall/building backgrounds.

"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ML_DIR       = Path(__file__).parent
CAMERA_TRAIN = ML_DIR / "datasets" / "security" / "camera_blind_spot" / "train"
# all camera types (Bullet=0, C-Mount=1, Dome=2, IR-Camera=3, Pan-Tilt-Zoom=4) → camera_blind_spot
CAMERA_CLASS_IDXS = {0, 1, 2, 3, 4}

BG_SOURCES = [
    ML_DIR / "datasets" / "condition" / "wall_crack"      / "train",
    ML_DIR / "datasets" / "condition" / "peeling_paint_1" / "train",
    ML_DIR / "datasets" / "condition" / "damp"            / "train",
]
OUT_DIR = ML_DIR / "datasets" / "security" / "camera_blind_spot_synthetic" / "train"

BG_SIZE      = 640
SCALE_MIN    = 0.04   # cameras are small objects on walls
SCALE_MAX    = 0.18
PATCH_PAD    = 0.06
MAX_BG_TRIES = 20


def _parse_boxes(label_path: Path, img_w: int, img_h: int):
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


def build_patch_pool(camera_train: Path) -> list:
    img_dir = camera_train / "images"
    lbl_dir = camera_train / "labels"
    patches = []
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    n = len(img_paths)
    print(f"  Building patch pool from {n} camera_blind_spot/train images …")

    for i, img_path in enumerate(img_paths):
        if i % 100 == 0:
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
            cls_idx = int(parts[0])
            if cls_idx not in CAMERA_CLASS_IDXS:
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
                if crop.shape[1] > 200:
                    sf = 200 / crop.shape[1]
                    crop = cv2.resize(crop, (200, max(8, int(crop.shape[0] * sf))), interpolation=cv2.INTER_AREA)
                patches.append(crop.copy())

    print(f"  Patch pool: {len(patches)} patches extracted.")
    return patches


def build_bg_list(sources: list) -> list:
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
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    boxes = _parse_boxes(lbl_path, w, h)
    crop_w, crop_h = min(w, BG_SIZE), min(h, BG_SIZE)

    for _ in range(MAX_BG_TRIES):
        x_off = random.randint(0, max(0, w - crop_w))
        y_off = random.randint(0, max(0, h - crop_h))
        if not any(_rects_overlap(x_off, y_off, x_off + crop_w, y_off + crop_h, *b) for b in boxes):
            crop = img[y_off:y_off + crop_h, x_off:x_off + crop_w]
            if crop_w < BG_SIZE or crop_h < BG_SIZE:
                crop = cv2.resize(crop, (BG_SIZE, BG_SIZE), interpolation=cv2.INTER_LINEAR)
            return crop

    x_off = random.randint(0, max(0, w - crop_w))
    y_off = random.randint(0, max(0, h - crop_h))
    crop  = img[y_off:y_off + crop_h, x_off:x_off + crop_w]
    if crop_w < BG_SIZE or crop_h < BG_SIZE:
        crop = cv2.resize(crop, (BG_SIZE, BG_SIZE), interpolation=cv2.INTER_LINEAR)
    return crop


def make_composite(bg: np.ndarray, patch: np.ndarray):
    bg_h, bg_w = bg.shape[:2]
    scale    = random.uniform(SCALE_MIN, SCALE_MAX)
    target_w = max(8, int(scale * bg_w))
    ph, pw   = patch.shape[:2]
    target_h = max(8, int(target_w * ph / pw))

    if target_w >= bg_w or target_h >= bg_h:
        return None, None, 0.0

    scaled  = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_AREA)
    alpha_j = random.uniform(0.75, 1.25)
    beta_j  = random.randint(-20, 20)
    scaled  = np.clip(scaled.astype(np.float32) * alpha_j + beta_j, 0, 255).astype(np.uint8)

    x1 = random.randint(0, bg_w - target_w)
    y1 = random.randint(0, bg_h - target_h)
    x2, y2 = x1 + target_w, y1 + target_h

    mask    = np.ones((target_h, target_w), dtype=np.float32)
    feather = max(1, min(target_h, target_w) // 6)
    ks      = feather * 2 + 1
    mask    = cv2.GaussianBlur(mask, (ks, ks), feather / 3.0)
    alpha   = mask[:, :, np.newaxis]

    comp    = bg.copy()
    roi     = comp[y1:y2, x1:x2].astype(np.float32)
    blended = alpha * scaled.astype(np.float32) + (1 - alpha) * roi
    comp[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    cx_n  = (x1 + x2) / 2 / bg_w
    cy_n  = (y1 + y2) / 2 / bg_h
    w_n   = target_w / bg_w
    h_n   = target_h / bg_h
    label = f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"
    return comp, label, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_img = OUT_DIR / "images"
    out_lbl = OUT_DIR / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Proptyze — Synthetic Camera Blind Spot Composites")
    print("=" * 60)
    print(f"  Target  : {args.target} composites")
    print(f"  Output  : {OUT_DIR}\n")

    patches = build_patch_pool(CAMERA_TRAIN)
    if not patches:
        raise SystemExit("ERROR: no camera patches extracted")

    bg_list = build_bg_list(BG_SOURCES)
    if not bg_list:
        raise SystemExit("ERROR: no background images found")

    generated, attempts = 0, 0
    while generated < args.target and attempts < args.target * 10:
        attempts += 1
        patch             = random.choice(patches)
        bg_path, lbl_path = random.choice(bg_list)
        bg                = get_bg_crop(bg_path, lbl_path)
        if bg is None:
            continue
        comp, label, scale = make_composite(bg, patch)
        if comp is None:
            continue
        cv2.imwrite(str(out_img / f"synth_camera_{generated:05d}.jpg"), comp, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_lbl / f"synth_camera_{generated:05d}.txt").write_text(label)
        generated += 1
        if generated % 100 == 0:
            print(f"    [{generated}/{args.target}]")

    print(f"\n  Done — {generated} composites saved to {OUT_DIR}")
    print(f"  Next: add camera_blind_spot_synthetic to merge_security.py, re-merge, train v2")


if __name__ == "__main__":
    main()
