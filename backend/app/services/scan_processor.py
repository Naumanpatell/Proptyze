import logging
import os
import uuid

import cv2

from sqlalchemy.orm import Session

from app.models.database import SessionLocal, Scan, Report
from app.services.frame_extractor import extract_frames
from app.services.blur_detection import is_blurry
from app.services.yolo_detector import detect
from app.services.scoring_engine import compute_score
from app.services.report_generator import generate_summary

logger = logging.getLogger(__name__)

FRAMES_BASE_DIR = os.path.join("uploads", "frames")


def _update_scan(db: Session, scan_id: str, **kwargs) -> None:
    db.query(Scan).filter(Scan.id == scan_id).update(kwargs)
    db.commit()

    
async def process_scan(scan_id: str, video_path: str | None = None, image_paths: list[str] | None = None) -> None:


    db = SessionLocal()
    try:
        _update_scan(db, scan_id, status="processing", stage="extracting_frames", progress=5)

        frames_dir = os.path.join(FRAMES_BASE_DIR, scan_id)
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths: list[str] = []

        # Video frames
        if video_path:
            logger.info("[%s] Extracting frames from video: %s", scan_id, video_path)
            video_frames = extract_frames(video_path, frames_dir, fps=1)
            logger.info("[%s] %d sharp video frames extracted", scan_id, len(video_frames))
            frame_paths.extend(video_frames)

        # Uploaded images added with the frames of the video, if any. Blurry images are discarded.
        if image_paths:
            logger.info("[%s] Adding %d uploaded images to frame pool", scan_id, len(image_paths))
            img_offset = len(frame_paths)
            for i, src in enumerate(image_paths):
                img = cv2.imread(src)
                if img is None:
                    logger.warning("[%s] Could not read image: %s", scan_id, src)
                    continue
                if is_blurry(img):
                    logger.debug("[%s] Uploaded image %d discarded — blurry", scan_id, i)
                    continue
                dest = os.path.join(frames_dir, f"photo_{img_offset + i:05d}.jpg")
                cv2.imwrite(dest, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_paths.append(os.path.abspath(dest))
            logger.info("[%s] Frame pool after images: %d total", scan_id, len(frame_paths))

        if not frame_paths:
            logger.warning("[%s] No usable frames — all blurry, corrupt, or empty input", scan_id)
            _update_scan(db, scan_id, status="failed", stage="extracting_frames", progress=0)
            return

        _update_scan(db, scan_id, stage="frames_ready", progress=25)

        # --- YOLO detection ---
        _update_scan(db, scan_id, stage="detecting", progress=30)
        logger.info("[%s] Running YOLO on %d frames", scan_id, len(frame_paths))
        detections = detect(frame_paths)
        logger.info("[%s] %d detections found", scan_id, len(detections))
        _update_scan(db, scan_id, progress=65)

        # --- Scoring ---
        _update_scan(db, scan_id, stage="scoring", progress=70)
        score_result = compute_score(detections, neighbourhood_data={}, frame_paths=frame_paths)
        score: int = score_result["score"]
        grade: str = score_result["grade"]
        logger.info("[%s] Score: %d  Grade: %s", scan_id, score, grade)

        # --- Report ---
        _update_scan(db, scan_id, stage="generating_report", progress=85)
        summary = await generate_summary(score, detections, neighbourhood={})
        report = Report(
            id=str(uuid.uuid4()),
            scan_id=scan_id,
            score=score,
            grade=grade,
            ai_summary=summary,
            detections=detections,
        )
        db.add(report)
        db.commit()

        _update_scan(db, scan_id, status="complete", stage="done", progress=100)
        logger.info("[%s] Pipeline complete — score %d (%s)", scan_id, score, grade)

    except Exception as exc:
        logger.exception("[%s] Pipeline failed: %s", scan_id, exc)
        _update_scan(db, scan_id, status="failed", stage="error", progress=0)

    finally:
        db.close()
