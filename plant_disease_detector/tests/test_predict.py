"""Tests for prediction CLI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import predict as predict_module
from predict import predict_image


def test_predict_image(
    tiny_image: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    label_map_path: Path,
    num_classes: int,
) -> None:
    out = predict_image(
        tiny_image,
        checkpoint_path,
        manifest_path=manifest_path,
        topk=min(3, num_classes),
        image_size=64,
    )
    assert "top_k" in out
    assert len(out["top_k"]) >= 1
    assert out["top1"]["class_id"] in range(num_classes)


def test_predict_main(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image: Path,
    checkpoint_path: Path,
    manifest_path: Path,
) -> None:
    monkeypatch.setenv("PLANT_DISEASE_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv("PLANT_DISEASE_MANIFEST", str(manifest_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--image",
            str(tiny_image),
            "--manifest-path",
            str(manifest_path),
            "--topk",
            "2",
        ],
    )
    predict_module.main()
