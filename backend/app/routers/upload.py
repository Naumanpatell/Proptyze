import os
import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.models.database import Scan, get_db
from app.services.scan_processor import process_scan

router = APIRouter(tags=["upload"])

VIDEO_EXTENSIONS = {"mp4", "mov", "avi"}
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024
MAX_IMAGES = 20


def _ext(filename: str) -> str:
    return (filename or "").rsplit(".", 1)[-1].lower()


@router.post("/upload", status_code=202)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(status_code=422, detail="No files provided.")

    exts = [_ext(f.filename) for f in files]
    videos = [(f, e) for f, e in zip(files, exts) if e in VIDEO_EXTENSIONS]
    images = [(f, e) for f, e in zip(files, exts) if e in IMAGE_EXTENSIONS]
    unknown = [f.filename for f, e in zip(files, exts) if e not in VIDEO_EXTENSIONS and e not in IMAGE_EXTENSIONS]

    if unknown:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type(s): {', '.join(unknown)}. "
                   "Accepted videos: MP4, MOV, AVI. Accepted images: JPG, PNG, WEBP, BMP, GIF.",
        )
    if len(videos) > 1:
        raise HTTPException(status_code=422, detail="Only one video file can be uploaded at a time.")
    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=422, detail=f"Max {MAX_IMAGES} photos per upload.")
    if not videos and not images:
        raise HTTPException(status_code=422, detail="No valid files provided.")

    # Read and size-check everything
    os.makedirs(settings.upload_dir, exist_ok=True)
    scan_id = str(uuid.uuid4())

    video_path = None
    if videos:
        f, ext = videos[0]
        contents = await f.read()
        if len(contents) > MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"'{f.filename}' exceeds the {settings.max_upload_size_mb} MB limit.")
        dest = os.path.join(settings.upload_dir, f"{scan_id}.{ext}")
        with open(dest, "wb") as fp:
            fp.write(contents)
        video_path = dest

    image_paths = []
    for i, (f, ext) in enumerate(images):
        contents = await f.read()
        if len(contents) > MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"'{f.filename}' exceeds the {settings.max_upload_size_mb} MB limit.")
        dest = os.path.join(settings.upload_dir, f"{scan_id}_img{i:03d}.{ext}")
        with open(dest, "wb") as fp:
            fp.write(contents)
        image_paths.append(dest)

    # Build a human-readable label for the scan
    if videos and images:
        label = f"{videos[0][0].filename} + {len(images)} photo{'s' if len(images) > 1 else ''}"
    elif videos:
        label = videos[0][0].filename
    else:
        label = images[0][0].filename if len(images) == 1 else f"{len(images)} photos"

    scan = Scan(id=scan_id, filename=label, status="queued", progress=0, stage="waiting")
    db.add(scan)
    db.commit()

    background_tasks.add_task(process_scan, scan_id, video_path, image_paths or None)

    return {"scan_id": scan_id, "status": "queued", "filename": label}
