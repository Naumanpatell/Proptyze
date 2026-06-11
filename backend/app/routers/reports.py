import os
import shutil

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.database import get_db, Scan, Report
from app.config import settings

router = APIRouter(tags=["reports"])

FRAMES_BASE_DIR = os.path.join("uploads", "frames")


@router.get("/scans/{scan_id}/status")
async def scan_status(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    result = {
        "scan_id": scan.id,
        "status": scan.status,
        "progress": scan.progress,
        "stage": scan.stage,
        "filename": scan.filename,
    }

    if scan.status == "complete":
        report = db.query(Report).filter(Report.scan_id == scan_id).first()
        if report:
            result["report_id"] = report.id

    return result


@router.get("/reports")
async def list_reports(db: Session = Depends(get_db)):
    reports = (
        db.query(Report)
        .order_by(Report.created_at.desc())
        .all()
    )
    items = []
    for r in reports:
        scan = db.query(Scan).filter(Scan.id == r.scan_id).first()
        items.append({
            "id": r.id,
            "scan_id": r.scan_id,
            "score": r.score,
            "grade": r.grade,
            "filename": scan.filename if scan else None,
            "created_at": r.created_at.isoformat(),
        })
    return {"reports": items, "total": len(items)}


@router.get("/reports/{report_id}")
async def get_report(report_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    scan = db.query(Scan).filter(Scan.id == report.scan_id).first()

    return {
        "id": report.id,
        "scan_id": report.scan_id,
        "score": report.score,
        "grade": report.grade,
        "ai_summary": report.ai_summary,
        "detections": report.detections or [],
        "filename": scan.filename if scan else None,
        "created_at": report.created_at.isoformat(),
    }


@router.delete("/reports/{report_id}")
async def delete_report(report_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    scan_id = report.scan_id
    db.delete(report)

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if scan:
        db.delete(scan)

    db.commit()

    # Clean up uploaded file and extracted frames
    for candidate in (os.listdir(settings.upload_dir) if os.path.isdir(settings.upload_dir) else []):
        if candidate.startswith(scan_id):
            try:
                os.remove(os.path.join(settings.upload_dir, candidate))
            except OSError:
                pass

    frames_dir = os.path.join(FRAMES_BASE_DIR, scan_id)
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)

    return {"deleted": report_id}


@router.get("/reports/{report_id}/pdf")
async def download_report_pdf(report_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    raise HTTPException(status_code=501, detail="PDF export not yet implemented")
