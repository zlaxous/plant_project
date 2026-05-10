"""Prepare PlantVillage/New Plant Diseases data into canonical manifest.

Path discovery confirmed by generate_manifest.bat:
  New Plant Diseases: .../New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train
  PlantVillage: .../PlantVillage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CANONICAL_CLASSES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

ALIASES: dict[str, str] = {
    "Pepper__bell___Bacterial_spot": "Pepper,_bell___Bacterial_spot",
    "Pepper__bell___healthy": "Pepper,_bell___healthy",
    "Tomato_Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato_Early_blight": "Tomato___Early_blight",
    "Tomato_Late_blight": "Tomato___Late_blight",
    "Tomato_Leaf_Mold": "Tomato___Leaf_Mold",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot": "Tomato___Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato_healthy": "Tomato___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust": "Corn_(maize)___Common_rust_",
    "Corn___Northern_Leaf_Blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn___healthy": "Corn_(maize)___healthy",
}


# ─── Filesystem utilities ───────────────────────────────────────────────────

_dir_cache: dict[str, list[str]] = {}

def _listdir(path: str) -> list[str]:
    if path in _dir_cache:
        return _dir_cache[path]
    root = Path(path)
    if not root.is_dir():
        _dir_cache[path] = []
        return []
    items = [p.name for p in root.iterdir() if p.is_dir()]
    _dir_cache[path] = sorted(items)
    return _dir_cache[path]


def _image_files(path: str) -> list[str]:
    root = Path(path)
    if not root.is_dir():
        return []
    items: list[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        items.extend(sorted(str(p.name) for p in root.glob(ext) if p.is_file()))
    return items


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_class_name(class_name: str) -> str | None:
    if class_name in CANONICAL_CLASSES:
        return class_name
    if class_name in ALIASES:
        return ALIASES[class_name]
    cn = class_name.strip().replace("__", "___")
    if cn in CANONICAL_CLASSES:
        return cn
    return None


def generate_health_report(samples: list[dict[str, Any]], threshold: int = 50) -> dict[str, Any]:
    split_counts = Counter(s["split"] for s in samples if "split" in s)
    class_counts: dict[str, int] = defaultdict(int)
    for s in samples:
        class_counts[s["class_name"]] += 1
    for c in CANONICAL_CLASSES:
        class_counts.setdefault(c, 0)
    low = sorted([n for n, c in class_counts.items() if c < threshold])
    return {
        "total_samples": len(samples),
        "split_counts": dict(split_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "low_count_classes": low,
        "low_count_threshold": threshold,
    }


def _find_dataset_root(base: Path, candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        candidate_path = base / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def gather_samples(root: str, source: str, rng: random.Random, seen: set[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    class_dirs = [d for d in _listdir(root) if d not in ('.', '..')]
    print(f"  Scanning {len(class_dirs)} dirs in {root}")
    for cd in class_dirs:
        cname = normalize_class_name(cd)
        if cname is None:
            continue
        full_dir = os.path.join(root, cd)
        imgs = _image_files(full_dir)
        if not imgs:
            continue
        for fname in imgs:
            fpath = os.path.join(full_dir, fname)
            try:
                h = compute_sha256(fpath)
            except Exception:
                continue
            if h in seen:
                continue
            seen.add(h)
            samples.append({
                "image_path": fpath,
                "class_name": cname,
                "class_id": CANONICAL_CLASSES.index(cname),
                "source": source,
                "sha256": h,
            })
    rng.shuffle(samples)
    return samples


def assign_splits(samples: list[dict[str, Any]], seed: int = 42) -> None:
    rng = random.Random(seed)
    by_class: dict[int, list[dict[str, Any]]] = {}
    for s in samples:
        by_class.setdefault(s["class_id"], []).append(s)
    for cs in by_class.values():
        rng.shuffle(cs)
        total = len(cs)
        if total <= 1:
            for s in cs:
                s["split"] = "train"
            continue
        if total == 2:
            tc = 1
            vc = 2
        else:
            val_count = max(1, int(total * 0.1))
            test_count = max(1, int(total * 0.1))
            if val_count + test_count >= total - 1:
                tc = 1
                val_count = 1
                test_count = total - 2
            else:
                tc = total - val_count - test_count
            if tc < 1:
                tc = 1
            vc = tc + val_count
        for i, s in enumerate(cs):
            s["split"] = "train" if i < tc else "val" if i < vc else "test"


def generate_synthetic_samples(
    package_root: Path,
    seed: int = 42,
    images_per_class: int = 20,
) -> list[dict[str, Any]]:
    """Create minimal random JPEGs for every canonical class (tests / bootstrap)."""
    np_rng = np.random.default_rng(seed)
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    syn_dir = package_root / "data" / "synthetic_images"
    syn_dir.mkdir(parents=True, exist_ok=True)
    for class_name in CANONICAL_CLASSES:
        cid = CANONICAL_CLASSES.index(class_name)
        cd = syn_dir / f"class_{cid:02d}"
        cd.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            arr = np_rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            ip = cd / f"img_{seed}_{cid}_{i}.jpg"
            Image.fromarray(arr).save(ip, quality=85)
            h = compute_sha256(str(ip))
            samples.append({
                "image_path": str(ip),
                "class_name": class_name,
                "class_id": cid,
                "source": "synthetic_bootstrap",
                "sha256": h,
            })
    rng.shuffle(samples)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-fallback", action="store_true")
    parser.add_argument("--synthetic-images-per-class", type=int, default=20)
    args = parser.parse_args()

    project_root = str(args.project_root)
    package_root = Path(args.project_root) / "plant_disease_detector"
    raw = Path(project_root) / "data" / "raw"

    np_base = _find_dataset_root(
        raw / "new_plant_disease",
        [
            Path("New Plant Diseases Dataset(Augmented)") / "New Plant Diseases Dataset(Augmented)",
            Path("New Plant Diseases Dataset(Augmented)"),
            Path("."),
        ],
    )
    pv_base = _find_dataset_root(
        raw / "plant_village",
        [
            Path("PlantVillage") / "PlantVillage",
            Path("PlantVillage"),
            Path("."),
        ],
    )

    if np_base is None:
        raise FileNotFoundError(
            f"Could not locate New Plant Diseases dataset under {raw / 'new_plant_disease'}"
        )
    if pv_base is None:
        raise FileNotFoundError(f"Could not locate PlantVillage dataset under {raw / 'plant_village'}")

    np_train_path = np_base / "train"
    np_valid_path = np_base / "valid"
    pv_path = pv_base

    print(f"New Plant Diseases base: {np_base}")
    print(f"New Plant Diseases train: {np_train_path}")
    print(f"New Plant Diseases valid: {np_valid_path} (exists={np_valid_path.exists()})")
    print(f"PlantVillage: {pv_path}")

    rng = random.Random(args.seed)
    seen: set[str] = set()
    np_train_samples = gather_samples(str(np_train_path), "vipoooool/new-plant-diseases-dataset", rng, seen)
    np_valid_samples = []
    if np_valid_path.exists():
        np_valid_samples = gather_samples(str(np_valid_path), "vipoooool/new-plant-diseases-dataset", rng, seen)
    else:
        print("Warning: New Plant Diseases valid/ folder not found; skipping valid source ingestion.")
    pv_samples = gather_samples(str(pv_path), "abdallahalidev/plantvillage-dataset", rng, seen)

    samples = np_train_samples + np_valid_samples + pv_samples
    print(f"Total samples: {len(samples)}")

    if len(samples) == 0 and args.synthetic_fallback:
        samples = generate_synthetic_samples(
            package_root,
            seed=args.seed,
            images_per_class=args.synthetic_images_per_class,
        )
        print(f"Generated {len(samples)} synthetic samples")

    assign_splits(samples, seed=args.seed)

    label_map = {str(i): c for i, c in enumerate(CANONICAL_CLASSES)}
    manifest = {
        "sources": ["abdallahalidev/plantvillage-dataset", "vipoooool/new-plant-diseases-dataset"],
        "num_classes": len(CANONICAL_CLASSES),
        "seed": args.seed,
        "samples": samples,
    }
    health = generate_health_report(samples)

    (package_root / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (package_root / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (package_root / "data" / "dataset_health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")

    print("\nDataset health report")
    print(json.dumps(health, indent=2))


if __name__ == "__main__":
    main()