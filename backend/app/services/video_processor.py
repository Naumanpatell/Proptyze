import logging
import os
import uuid

from sqlalchemy.orm import Session

from app.models.database import SessionLocal, Scan, Report
from app.services.frame_extractor import extract_frames
from app.services.yolo_detector import detect
from app.services.scoring_engine import compute_score
from app.services.report_generator import generate_summary

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
        logger.info("[%s] Running YOLO detection on %d frames", scan_id, len(frame_paths))
        detections = detect(frame_paths)
        logger.info("[%s] %d total detections found", scan_id, len(detections))
        _update_scan(db, scan_id, progress=65)

        # scoring
        _update_scan(db, scan_id, stage="scoring", progress=70)
        score_result = compute_score(detections, neighbourhood_data={}, frame_paths=frame_paths)
        score: int = score_result["score"]
        grade: str = score_result["grade"]
        logger.info("[%s] Score: %d  Grade: %s", scan_id, score, grade)

        # report generation
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
