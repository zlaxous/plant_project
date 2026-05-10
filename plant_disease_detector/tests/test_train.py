"""Tests for training utilities (short runs, no weight download)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import train as train_module
from model import build_model
from train import EarlyStopping, TrainConfig, _accuracy, _subset_if_needed, train


def test_early_stopping_triggers() -> None:
    es = EarlyStopping(patience=2)
    assert es.step(0.5) is False
    assert es.step(0.6) is False
    assert es.step(0.4) is False
    assert es.step(0.3) is True


def test_accuracy() -> None:
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    assert _accuracy(logits, targets) == pytest.approx(1.0)


def test_subset_if_needed_full_dataset(manifest_path: Path) -> None:
    from dataset import PlantDiseaseDataset, build_transforms

    ds = PlantDiseaseDataset.from_manifest(
        manifest_path, split="train", transform=build_transforms(32, train=False)
    )
    out = _subset_if_needed(ds, 1.0)
    assert out is ds


def test_train_short_run(
    monkeypatch: pytest.MonkeyPatch,
    manifest_path: Path,
    num_classes: int,
    tmp_path: Path,
) -> None:
    """Two-phase train without downloading ImageNet weights."""

    def _fake_build(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
        return build_model(num_classes, pretrained=False)

    monkeypatch.setattr(train_module, "build_model", _fake_build)

    cfg = TrainConfig(
        manifest_path=manifest_path,
        output_dir=tmp_path / "ckpt",
        image_size=32,
        batch_size=2,
        num_workers=0,
        warmup_epochs=1,
        finetune_epochs=1,
        patience=2,
        subset_fraction=1.0,
    )
    ckpt = train(cfg, num_classes=num_classes)
    assert ckpt.is_file()
