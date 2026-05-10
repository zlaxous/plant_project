"""Shared fixtures for plant_disease_detector tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from model import build_model


@pytest.fixture()
def num_classes() -> int:
    return 4


@pytest.fixture()
def tiny_image(tmp_path: Path) -> Path:
    path = tmp_path / "leaf.jpg"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture()
def label_map_path(tmp_path: Path, num_classes: int) -> Path:
    names = [f"class_{i}" for i in range(num_classes)]
    mapping = {str(i): names[i] for i in range(num_classes)}
    p = tmp_path / "label_map.json"
    p.write_text(json.dumps(mapping), encoding="utf-8")
    return p


@pytest.fixture()
def manifest_path(tmp_path: Path, num_classes: int) -> Path:
    """Minimal manifest with one sample per split per class."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    samples: list[dict] = []
    for cid in range(num_classes):
        for split, suffix in [("train", "tr"), ("val", "v"), ("test", "te")]:
            arr = torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8).numpy()
            ip = img_dir / f"c{cid}_{split}_{suffix}.jpg"
            Image.fromarray(arr).save(ip)
            samples.append(
                {
                    "image_path": str(ip.resolve()),
                    "class_name": f"class_{cid}",
                    "class_id": cid,
                    "source": "test",
                    "sha256": "0" * 64,
                    "split": split,
                }
            )
    manifest = {
        "sources": ["test"],
        "num_classes": num_classes,
        "seed": 1,
        "samples": samples,
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return p


@pytest.fixture()
def checkpoint_path(tmp_path: Path, num_classes: int) -> Path:
    model = build_model(num_classes=num_classes, pretrained=False)
    path = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict(), "val_acc": 0.5}, path)
    return path
