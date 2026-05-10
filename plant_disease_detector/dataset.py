"""Dataset utilities for the Plant Disease Detector project.

Uses subprocess-based file I/O to handle Windows paths with parentheses,
which are a known issue with os.path/pathlib on certain Windows builds.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class SampleRecord:
    """Represents one dataset sample entry from manifest."""

    image_path: str
    class_id: int
    class_name: str
    split: str
    source: str
    sha256: str


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for exact duplicate detection."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_image_safe(path: str) -> Image.Image:
    """Load an image from path, falling back to cmd.exe copy if direct open fails.
    
    Handles paths containing parentheses on Windows by copying to a temp file
    via cmd.exe if needed.
    """
    try:
        return Image.open(path).convert("RGB")
    except (OSError, FileNotFoundError, PermissionError):
        pass
    
    # Fallback: copy via cmd.exe /c copy (handles () paths)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        subprocess.run(
            ["cmd", "/c", "copy", path, tmp_path, "/y"],
            capture_output=True, timeout=30,
        )
        img = Image.open(tmp_path).convert("RGB")
        return img
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def build_transforms(image_size: int = 380, train: bool = True) -> Callable[[Image.Image], Any]:
    """Build torchvision transforms for train/eval pipelines."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.75, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                transforms.RandomErasing(
                    p=0.2,
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value="random",
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class PlantDiseaseDataset(Dataset[tuple[Any, int]]):
    """PyTorch dataset backed by `data/manifest.json`."""

    def __init__(
        self,
        records: list[SampleRecord],
        transform: Callable[[Image.Image], Any] | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.records = records
        self.transform = transform
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        split: str,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> PlantDiseaseDataset:
        """Create dataset from manifest for one split."""
        manifest_path = Path(manifest_path).resolve()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train/val/test")

        base_dir = manifest_path.parent.parent.parent

        selected: list[SampleRecord] = []
        for sample in data.get("samples", []):
            if sample.get("split") == split:
                selected.append(
                    SampleRecord(
                        image_path=sample["image_path"],
                        class_id=int(sample["class_id"]),
                        class_name=sample["class_name"],
                        split=sample["split"],
                        source=sample.get("source", ""),
                        sha256=sample.get("sha256", ""),
                    )
                )
        return cls(records=selected, transform=transform, base_dir=base_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self.records[index]
        image_path = str(self.base_dir / record.image_path)
        image = load_image_safe(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, record.class_id


def generate_health_report(
    samples: Iterable[dict[str, Any]],
    low_count_threshold: int = 50,
    all_classes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Generate dataset health stats including low-count classes."""
    samples_list = list(samples)
    split_counts = Counter(sample["split"] for sample in samples_list)
    class_counts: dict[str, int] = defaultdict(int)
    for sample in samples_list:
        class_counts[sample["class_name"]] += 1
    if all_classes is not None:
        for class_name in all_classes:
            class_counts.setdefault(class_name, 0)
    low_count_classes = sorted(
        [name for name, count in class_counts.items() if count < low_count_threshold]
    )
    return {
        "total_samples": len(samples_list),
        "split_counts": dict(split_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "low_count_classes": low_count_classes,
        "low_count_threshold": low_count_threshold,
    }