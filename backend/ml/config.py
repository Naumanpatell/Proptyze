# Detection classes

CONDITION_CLASSES = [
    "wall_crack",       # cracksdetection — crackdetection-xedoq — v1 — 956 images
    "damp",             # damp — wall-quality-detection — dip-project-f2lk5 — 250 imgs — v1 
    "mould",            # mould-0hu11 — yolo-wislr — v2 — 6,233 images — mAP 95.5%
    "damaged_flooring", # could not find any
    "peeling_paint",    # 4-jwvne — building-defect-vbgco — v1 — 27 images
                        # paint_defects_ds2 — train-te25s — v4 — 1.2k images
    "broken_fixture",
]

SECURITY_CLASSES = [
    "weak_entry",
    "poor_lighting",
    "fence_gap",
    "blocked_exit",
    "camera_blind_spot",
]

NUM_CONDITION_CLASSES = len(CONDITION_CLASSES)
NUM_SECURITY_CLASSES = len(SECURITY_CLASSES)

CONDITION_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CONDITION_CLASSES)}
SECURITY_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SECURITY_CLASSES)}

CONDITION_IDX_TO_CLASS = {idx: cls for cls, idx in CONDITION_CLASS_TO_IDX.items()}
SECURITY_IDX_TO_CLASS = {idx: cls for cls, idx in SECURITY_CLASS_TO_IDX.items()}
