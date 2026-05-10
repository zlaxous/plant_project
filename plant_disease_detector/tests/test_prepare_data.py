"""Tests for synthetic data bootstrap."""

from __future__ import annotations

from pathlib import Path

from prepare_data import assign_splits, generate_synthetic_samples


def test_assign_splits_sets_three_partitions() -> None:
    samples = [{"class_id": 0, "foo": 1} for _ in range(20)]
    assign_splits(samples, seed=0)
    splits = {s["split"] for s in samples}
    assert splits <= {"train", "val", "test"}


def test_generate_synthetic_samples_creates_images(tmp_path: Path) -> None:
    pkg = tmp_path / "plant_disease_detector"
    (pkg / "data").mkdir(parents=True)
    samples = generate_synthetic_samples(pkg, seed=7, images_per_class=2)
    assert len(samples) > 0
    first = Path(samples[0]["image_path"])
    assert first.is_file()
