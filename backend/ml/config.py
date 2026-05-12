CONDITION_CLASSES = [
    "wall_crack",       # dataset: cracksdetection | workspace: crackdetection-xedoq | v1 | 956 images
    "damp",             # dataset: wall-quality-detection | workspace: dip-project-f2lk5 | v1 | 250 images
    "mould",            # dataset: mould-0hu11 | workspace: yolo-wislr | v2 | 6,233 images | mAP 95.5%
    "peeling_paint",    # dataset: 4-jwvne | workspace: building-defect-vbgco | v1 | 27 images
                        # dataset: paint_defects_ds2 | workspace: train-te25s | v4 | 1,120 images
    "broken_fixture",   # no public dataset found -- shoot your own -- target 60-80 images

    # "damaged_flooring"  -- DROPPED for MVP -- no UK-relevant dataset exists (all datasets
    #                        are industrial/concrete, not UK residential carpet or wood flooring)
    #                        revisit Month 2 -- shoot your own in real UK properties
]

SECURITY_CLASSES = [
    "weak_entry",        # dataset: door-lock-srtfa | workspace: kirby-x1yw0 | v2 | 88 images
                         # dataset: door-3oaz9 | workspace: door-ca23l | v1 | 202 images | mAP 99.5%
    "fence_gap",         # dataset: broken-fence | workspace: TBD | v2 | 1,208 images | mAP 85.2%
                         # workspace slug still needed -- check URL on Roboflow page
    # "blocked_exit",      # DROPPED for MVP -- no public dataset exists for blocked or
    #                         obstructed doorways. Revisit Month 2 -- shoot your own.
    #                         Target: 30-50 images of doors blocked by boxes or furniture
    "camera_blind_spot", # dataset: cctv-camera-detection-ugbhl | workspace: ajmairworkspace | v3
    # "poor_lighting"     -- DROPPED as a CV class -- poor lighting is a property of the whole
    #                        image, not a detectable object. Replaced with brightness check in
    #                        the scoring engine using average Laplacian variance per frame
]

NUM_CONDITION_CLASSES = len(CONDITION_CLASSES)
NUM_SECURITY_CLASSES = len(SECURITY_CLASSES)

CONDITION_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CONDITION_CLASSES)}
SECURITY_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SECURITY_CLASSES)}

CONDITION_IDX_TO_CLASS = {idx: cls for cls, idx in CONDITION_CLASS_TO_IDX.items()}
SECURITY_IDX_TO_CLASS = {idx: cls for cls, idx in SECURITY_CLASS_TO_IDX.items()}