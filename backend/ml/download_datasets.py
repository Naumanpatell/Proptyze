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
        "datasets/condition/wall_crack"        # 956 images
    ),
    (
        "yolo-wislr",
        "mould-0hu11",
        2,
        "datasets/condition/mould"             # 6,233 images | mAP 95.5%
    ),
    (
        "building-defect-vbgco",
        "4-jwvne",
        1,
        "datasets/condition/peeling_paint_1"   # 27 images | high quality
    ),
    (
        "train-te25s",
        "paint_defects_ds2",
        4,
        "datasets/condition/peeling_paint_2"   # 1,120 images
    ),
    (
        "dip-project-f2lk5",
        "wall-quality-detection",
        1,
        "datasets/condition/damp"              # 250 images | classes: crack, damp, fungus
    ),
    # broken_fixture -- no public dataset -- shoot your own -- target 60-80 images
    # damaged_flooring -- DROPPED for MVP -- no UK-relevant dataset exists

    # ── SECURITY MODEL ───────────────────────────────────────────────
    (
        "kirby-x1yw0",
        "door-lock-srtfa",
        2,
        "datasets/security/weak_entry_locks"   # 88 images | mAP 92%
    ),
    (
        "door-ca23l",
        "door-3oaz9",
        1,
        "datasets/security/weak_entry_doors"   # 202 images | mAP 99.5%
    ),
    (
        "ajmairworkspace",
        "cctv-camera-detection-ugbhl",
        3,
        "datasets/security/camera_blind_spot"  # real CCTV cameras on buildings
    ),
    # fence_gap -- dataset found (broken-fence | 1,208 images | mAP 85.2%) -- workspace slug TBD
    # blocked_exit -- DROPPED for MVP -- no public dataset exists
    # poor_lighting -- DROPPED as CV class -- handled via brightness check in scoring engine
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
        project = rf.workspace(workspace_slug).project(project_slug)
        version = project.version(version_num)
        version.download("yolov8", location=output_path, overwrite=True)
        print(f"Done -> {output_path}")

    except Exception as e:
        print(f"Failed: {project_slug}")
        print(f"   Error: {e}")
        print(f"   -> Check the workspace/project slug at universe.roboflow.com")

print("\n\nStatus summary:")

print("  DOWNLOADED:")
print("    datasets/condition/wall_crack")
print("    datasets/condition/mould")
print("    datasets/condition/peeling_paint_1")
print("    datasets/condition/peeling_paint_2")
print("    datasets/condition/damp")
print("    datasets/security/weak_entry_locks")
print("    datasets/security/weak_entry_doors")
print("    datasets/security/camera_blind_spot")
print("")
print("  PENDING:")
print("    datasets/security/fence_gap     -- workspace slug needed for broken-fence dataset")
print("    datasets/condition/broken_fixture -- shoot your own, target 60-80 images")
print("")
print("  DROPPED FOR MVP:")
print("    damaged_flooring -- no UK-relevant dataset, revisit Month 2")
print("    blocked_exit     -- no public dataset, revisit Month 2")
print("    poor_lighting    -- replaced with brightness check in scoring engine")