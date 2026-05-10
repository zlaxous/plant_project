"""Tests for path helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paths import load_label_map, load_manifest, num_classes_from_manifest


def test_num_classes_from_manifest(manifest_path: Path, num_classes: int) -> None:
    assert num_classes_from_manifest(manifest_path) == num_classes


def test_load_manifest(manifest_path: Path) -> None:
    data = load_manifest(manifest_path)
    assert "samples" in data


def test_load_label_map_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "lm.json"
    p.write_text(json.dumps({"0": "a", "1": "b"}), encoding="utf-8")
    m = load_label_map(p)
    assert m[0] == "a"


def test_label_map_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_label_map(tmp_path / "nope.json")
