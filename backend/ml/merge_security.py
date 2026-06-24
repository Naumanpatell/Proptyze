"""
merge_security.py — Merge all security datasets into one unified dataset
Remaps each source dataset's class IDs to match SECURITY_CLASSES.

Run from: backend/ml/
    python merge_security.py

Run this before train.py --model security
"""

import shutil
import yaml
from pathlib import Path

ML_DIR       = Path(__file__).parent
DATASETS_DIR = ML_DIR / "datasets" / "security"
MERGED_DIR   = DATASETS_DIR / "merged"

SECURITY_CLASSES = ["weak_entry", "fence_gap", "camera_blind_spot"]
TARGET_IDX       = {cls: i for i, cls in enumerate(SECURITY_CLASSES)}

# Keys must match folder names under datasets/security/
# Values: { original_class_name_lower → target_class | None }
#   None  = discard annotations for that source class
#   "*"   = catch-all for any class not explicitly listed
DATASET_CLASS_MAPS: dict[str, dict[str, str | None]] = {
    "weak_entry_locks": {
        "*": "weak_entry",
    },
    "weak_entry_doors": {
        "*": "weak_entry",
    },
    "camera_blind_spot": {
        "*": "camera_blind_spot",
    },
    "camera_blind_spot_synthetic": {
        "*": "camera_blind_spot",
    },
    "weak_entry_synthetic": {
        "*": "weak_entry",
    },
    # fence_gap: workspace slug still TBD — add entry here when dataset is confirmed
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS   = ["train", "valid", "test"]


def _resolve_split_dir(data_yaml: dict, split: str, dataset_dir: Path) -> Path | None:
    keys = ["val", "valid"] if split == "valid" else [split]
    for key in keys:
        raw = data_yaml.get(key)
        if not raw:
            continue
        p = Path(str(raw))
        if p.is_absolute():
            if p.exists():
                return p
            tail = Path(*p.parts[-2:]) if len(p.parts) >= 2 else Path(p.name)
            candidate = dataset_dir / tail
            if candidate.exists():
                return candidate
        else:
            candidate = dataset_dir / p
            if candidate.exists():
                return candidate
            # Roboflow exports use ../split/images relative to yaml; strip leading '..'
            stripped = Path(*[part for part in p.parts if part != '..'])
            candidate2 = dataset_dir / stripped
            if candidate2.exists():
                return candidate2
    return None


def _build_id_map(src_names: list[str], class_map: dict[str, str | None]) -> dict[int, int | None]:
    id_map: dict[int, int | None] = {}
    for src_idx, src_name in enumerate(src_names):
        target_cls = class_map.get(src_name.lower(), class_map.get("*"))
        id_map[src_idx] = TARGET_IDX.get(target_cls) if target_cls else None
    return id_map


def _remap_label(src: Path, dst: Path, id_map: dict[int, int | None]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        dst.write_text("")
        return
    kept = []
    for line in src.read_text().strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        new_cls = id_map.get(int(parts[0]))
        if new_cls is not None:
            kept.append(f"{new_cls} " + " ".join(parts[1:]))
    dst.write_text("\n".join(kept) + ("\n" if kept else ""))


def merge_one(dataset_name: str, class_map: dict[str, str | None]) -> dict[str, int]:
    src_dir   = DATASETS_DIR / dataset_name
    yaml_path = src_dir / "data.yaml"

    if not src_dir.exists():
        print(f"  [MISSING] {dataset_name}/ — run download_datasets.py first")
        return {}
    if not yaml_path.exists():
        print(f"  [SKIP] {dataset_name}: no data.yaml found")
        return {}

    with open(yaml_path) as f:
        data_yaml = yaml.safe_load(f)

    src_names: list[str] = data_yaml.get("names", [])
    if not src_names:
        print(f"  [SKIP] {dataset_name}: data.yaml has no 'names' field")
        return {}

    id_map      = _build_id_map(src_names, class_map)
    mapping_str = {
        src_names[i]: SECURITY_CLASSES[v] if v is not None else "DISCARD"
        for i, v in id_map.items()
    }
    print(f"  {dataset_name}: {mapping_str}")

    counts: dict[str, int] = {}
    for split in SPLITS:
        img_dir = _resolve_split_dir(data_yaml, split, src_dir)
        if not img_dir:
            continue

        lbl_dir = img_dir.parent.parent / split / "labels"
        if not lbl_dir.exists():
            lbl_dir = img_dir.parent / "labels"

        dst_img = MERGED_DIR / split / "images"
        dst_lbl = MERGED_DIR / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n = 0
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            stem = f"{dataset_name}__{img_path.stem}"
            shutil.copy2(img_path, dst_img / (stem + img_path.suffix))
            _remap_label(lbl_dir / (img_path.stem + ".txt"), dst_lbl / (stem + ".txt"), id_map)
            n += 1

        counts[split] = n

    return counts


def merge() -> Path:
    if MERGED_DIR.exists():
        print(f"Clearing {MERGED_DIR}")
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True)

    totals: dict[str, int] = {}
    for dataset_name, class_map in DATASET_CLASS_MAPS.items():
        counts = merge_one(dataset_name, class_map)
        for split, n in counts.items():
            totals[split] = totals.get(split, 0) + n

    print(f"\nTotal images: {totals}")

    if not totals.get("train"):
        print("\nERROR: No training images — run download_datasets.py first.")
        raise SystemExit(1)

    yaml_out  = MERGED_DIR / "data.yaml"
    yaml_data = {
        "path":  str(MERGED_DIR.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "nc":    len(SECURITY_CLASSES),
        "names": SECURITY_CLASSES,
    }
    if (MERGED_DIR / "test" / "images").exists():
        yaml_data["test"] = "test/images"

    with open(yaml_out, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    print(f"data.yaml -> {yaml_out}")
    return yaml_out


if __name__ == "__main__":
    print("=" * 60)
    print("  Merging security datasets")
    print("=" * 60)
    print(f"  Classes : {SECURITY_CLASSES}\n")
    merge()
    print("\nDone. Run: python train.py --model security")
