# download_datasets.py
# Location: backend/ml/download_datasets.py
# Run from backend/ml/ directory
# pip install roboflow

from roboflow import Roboflow
import os

API_KEY = "FdxByRCKD9r0CChZZ4dZ"

rf = Roboflow(api_key=API_KEY)

DATASETS = [
    # (workspace_slug, project_slug, version, output_path)

    # ── CONDITION MODEL ──────────────────────────────────────────────
    (
        "crackdetection-xedoq",
        "cracksdetection",
        1,
        "datasets/condition/wall_crack"
    ),
    (
        "yolo-wislr",
        "mould-0hu11",
        2,
        "datasets/condition/mould"
    ),
    (
        "building-defect-vbgco",
        "4-jwvne",
        1,
        "datasets/condition/peeling_paint_1"
    ),
    (
        "train-te25s",
        "paint_defects_ds2",
        4,
        "datasets/condition/peeling_paint_2"
    ),
    (
        "dip-project-f2lk5",
        "wall-quality-detection",
        1,
        "datasets/condition/damp"
    ),

    # ── SECURITY MODEL ───────────────────────────────────────────────
    # Add security datasets here as you confirm them
]

for workspace_slug, project_slug, version_num, output_path in DATASETS:
    print(f"\n{'='*60}")
    print(f"Checking: {project_slug}")
    print(f"{'='*60}")

    # Skip if folder already exists and has files in it
    if os.path.exists(output_path) and any(
        files for _, _, files in os.walk(output_path)
    ):
        print(f"Skipping — already downloaded -> {output_path}")
        continue

    print(f"Downloading: {project_slug} -> {output_path}")

    try:
        os.makedirs(output_path, exist_ok=True)
        project = rf.workspace(workspace_slug).project(project_slug)
        version = project.version(version_num)
        version.download("yolov8", location=output_path)
        print(f"Done -> {output_path}")

    except Exception as e:
        print(f"Failed: {project_slug}")
        print(f"   Error: {e}")
        print(f"   -> Check the workspace/project slug at universe.roboflow.com")

print("\n\nDone! Classes still needing datasets:")
