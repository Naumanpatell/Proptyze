import logging
import os

from sqlalchemy.orm import Session

from app.models.database import SessionLocal, Scan
from app.services.frame_extractor import extract_frames

logger = logging.getLogger(__name__)

# Frames land in  uploads/frames/<scan_id>/
FRAMES_BASE_DIR = os.path.join("uploads", "frames")


def _update_scan(db: Session, scan_id: str, **kwargs) -> None:
    db.query(Scan).filter(Scan.id == scan_id).update(kwargs)
    db.commit()


async def process_video(scan_id: str, video_path: str) -> None:
    """
    Entry point called by FastAPI BackgroundTasks.
    Each stage updates the scan record so the UI can show live progress.
    """
    db = SessionLocal()
    try:
        # frame extraction 
        _update_scan(db, scan_id, status="processing", stage="extracting_frames", progress=5)
        logger.info("[%s] Starting frame extraction from %s", scan_id, video_path)

        frames_dir = os.path.join(FRAMES_BASE_DIR, scan_id)
        frame_paths = extract_frames(video_path, frames_dir, fps=1)

        if not frame_paths:
            logger.warning("[%s] No usable frames extracted — video may be too blurry or corrupt", scan_id)
            _update_scan(db, scan_id, status="failed", stage="extracting_frames", progress=0)
            return

        logger.info("[%s] %d sharp frames saved to %s", scan_id, len(frame_paths), frames_dir)
        _update_scan(db, scan_id, stage="frames_ready", progress=25)

        # YOLO detection
        _update_scan(db, scan_id, stage="detecting", progress=30)
        # detections = run_yolo(frame_paths)

        # scoring
        _update_scan(db, scan_id, stage="scoring", progress=70)
        # score, grade = score_property(detections)

        # report generation
        _update_scan(db, scan_id, stage="generating_report", progress=85)
        # report = generate_report(scan_id, score, grade, detections)

        # Done
        _update_scan(db, scan_id, status="complete", stage="done", progress=100)
        logger.info("[%s] Pipeline complete", scan_id)

    except Exception as exc:
        logger.exception("[%s] Pipeline failed: %s", scan_id, exc)
        _update_scan(db, scan_id, status="failed", stage="error", progress=0)

    finally:
        db.close()
