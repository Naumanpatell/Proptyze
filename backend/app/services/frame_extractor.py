import os
import logging

import cv2

from app.services.blur_detection import is_blurry

logger = logging.getLogger(__name__)


def extract_frames( video_path: str, output_dir: str, fps: int = 1,) -> list[str]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_interval: int = max(1, round(video_fps / fps))

        logger.info(
            "Extracting from %s  |  video_fps=%.2f  interval=%d  total_frames=%d",
            os.path.basename(video_path), video_fps, frame_interval, total_frames,
        )

        saved_paths: list[str] = []
        frame_idx: int = 0
        sample_num: int = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames that fall on our sample cadence
            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / video_fps

                if is_blurry(frame):
                    logger.debug("Frame %d (t=%.2fs) discarded — blurry", frame_idx, timestamp_sec)
                else:
                    filename = f"frame_{sample_num:05d}_t{timestamp_sec:.2f}s.jpg"
                    out_path = os.path.join(output_dir, filename)
                    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_paths.append(os.path.abspath(out_path))
                    logger.debug("Saved %s", filename)

                sample_num += 1

            frame_idx += 1

        logger.info(
            "Done — %d/%d samples kept after blur filter",
            len(saved_paths), sample_num,
        )
        return saved_paths

    finally:
        cap.release()
