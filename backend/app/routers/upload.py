import os
import uuid
import shutil

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.models.database import Scan, get_db
from app.services.video_processor import process_video

router = APIRouter(tags=["upload"])


ALLOWED_EXTENSIONS = {"mp4", "mov", "avi"}

MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024


@router.post("/upload", status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Files Validation using the allowed extentions

    ext = (file.filename or "").rsplit(".", 1)[-1].lower() 
    # if there is no name used in file.filename then name will be taken as NONE hence "" protects the code from crashing the .rsplit() function

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension '.{ext}'. Accepted: .mp4, .mov, .avi.",
        )

    # Check the file size (Reads the whole file into the memory)
    contents = await file.read() # contents is just a sequence of bytes
    if len(contents) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_size_mb}MB size limit.",
        )

    # generate a random 128 bit number as a UUID
    os.makedirs(settings.upload_dir, exist_ok=True)
    scan_id = str(uuid.uuid4())

    # Safely builds the full path with the file name and the extention.
    dest_filename = f"{scan_id}.{ext}"
    dest_path = os.path.join(settings.upload_dir, dest_filename)
    # e.g. (uploads/a3f7c92e-4b1d-4e8a-9f3c-2d1e7b6a0c5f.mp4)
    with open(dest_path, "wb") as f:
        f.write(contents)

    # Create scan record in the database
    scan = Scan(
        id=scan_id,
        filename=file.filename,
        status="queued",
        progress=0,
        stage="waiting",
    )
    db.add(scan)
    db.commit()

    # Kick off the processing pipeline as a background task
    background_tasks.add_task(process_video, scan_id, dest_path)
  
    return {
        "scan_id": scan_id,
        "status": "queued",
        "filename": file.filename,
    }
