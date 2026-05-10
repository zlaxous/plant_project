"""Default filesystem paths and manifest helpers for the detector package."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent


def default_manifest_path() -> Path:
    """Return the canonical manifest path under this package."""
    env = os.environ.get("PLANT_DISEASE_MANIFEST", "").strip()
    if env:
        return Path(env)
    return PACKAGE_ROOT / "data" / "manifest.json"


def default_label_map_path() -> Path:
    """Return path to class index → name JSON."""
    env = os.environ.get("PLANT_DISEASE_LABEL_MAP", "").strip()
    if env:
        return Path(env)
    return PACKAGE_ROOT / "label_map.json"


def default_checkpoint_path() -> Path:
    """Checkpoint path from env or package default."""
    env = os.environ.get("PLANT_DISEASE_CHECKPOINT", "").strip()
    if env:
        return Path(env)
    return PACKAGE_ROOT / "checkpoints" / "best.pt"


def default_results_dir() -> Path:
    """Directory for evaluation metrics and plots."""
    return PACKAGE_ROOT / "results"


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load manifest JSON."""
    path = Path(manifest_path)
    return json.loads(path.read_text(encoding="utf-8"))


def num_classes_from_manifest(manifest_path: str | Path) -> int:
    """Read num_classes from manifest (defaults to 38)."""
    data = load_manifest(manifest_path)
    return int(data.get("num_classes", 38))


def load_label_map(label_map_path: str | Path | None = None) -> dict[int, str]:
    """Load label map as int → class name."""
    path = Path(label_map_path) if label_map_path is not None else default_label_map_path()
    raw: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}
