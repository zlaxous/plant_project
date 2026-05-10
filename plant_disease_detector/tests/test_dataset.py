"""Tests for dataset utilities."""

from __future__ import annotations

from pathlib import Path

from dataset import PlantDiseaseDataset, build_transforms, compute_sha256, generate_health_report


def test_build_transforms_returns_callable() -> None:
    tr = build_transforms(image_size=224, train=True)
    ev = build_transforms(image_size=224, train=False)
    assert callable(tr)
    assert callable(ev)


def test_plant_dataset_from_manifest(manifest_path: Path) -> None:
    ds = PlantDiseaseDataset.from_manifest(
        manifest_path, split="train", transform=build_transforms(64, train=False)
    )
    assert len(ds) >= 1
    x, y = ds[0]
    assert isinstance(y, int)
    assert x.shape[0] == 3


def test_compute_sha256_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    h = compute_sha256(p)
    assert len(h) == 64


def test_generate_health_report() -> None:
    samples = [
        {"split": "train", "class_name": "a"},
        {"split": "val", "class_name": "a"},
    ]
    rep = generate_health_report(samples, low_count_threshold=5, all_classes=["a", "b"])
    assert rep["total_samples"] == 2
    assert "b" in rep["class_counts"]
