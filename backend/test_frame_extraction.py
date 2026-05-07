import sys
import os
import logging

import cv2

# Make sure backend/app is importable when running from backend/
sys.path.insert(0, os.path.dirname(__file__))

from app.services.blur_detection import laplacian_variance, DEFAULT_BLUR_THRESHOLD
from app.services.frame_extractor import extract_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
BLUR_THRESHOLD = DEFAULT_BLUR_THRESHOLD   # lower = keep more frames
OUTPUT_DIR     = "test_output/frames"
FPS_SAMPLE     = 1                        # frames to extract per second

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python test_frame_extraction.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"Error: file not found — {video_path}")
        sys.exit(1)

    print(f"\nVideo       : {video_path}")
    print(f"Sample rate : {FPS_SAMPLE} fps")
    print(f"Blur cutoff : Laplacian variance < {BLUR_THRESHOLD}  (lower = keep more frames)")
    print(f"Output dir  : {OUTPUT_DIR}\n")

    # ── Per-frame sharpness report ────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval   = max(1, round(video_fps / FPS_SAMPLE))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total / video_fps

    print(f"{'Sample':>7}  {'Time (s)':>9}  {'Lap.Var':>10}  {'Status'}")
    print("-" * 45)

    sample_num = 0
    frame_idx  = 0
    kept = 0
    dropped = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            t   = frame_idx / video_fps
            lap = laplacian_variance(frame)
            status = "SHARP" if lap >= BLUR_THRESHOLD else "blurry"
            if lap >= BLUR_THRESHOLD:
                kept += 1
            else:
                dropped += 1
            print(f"{sample_num:>7}  {t:>9.2f}  {lap:>10.1f}  {status}")
            sample_num += 1
        frame_idx += 1

    cap.release()

    # ── Extract and save ──────────────────────────────────────────────────────
    print(f"\nSummary: {kept} sharp  |  {dropped} blurry  |  {sample_num} total samples")
    print(f"Video duration: {duration_s:.1f}s\n")

    print(f"Saving sharp frames to {OUTPUT_DIR} ...")
    saved = extract_frames(video_path, OUTPUT_DIR, fps=FPS_SAMPLE)
    print(f"\nDone — {len(saved)} frames saved.\n")

    for p in saved:
        print(" ", p)


if __name__ == "__main__":
    main()
