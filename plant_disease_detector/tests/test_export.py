"""Tests for ONNX / model export."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import export as export_module


def test_verify_onnx_matches_torch(
    checkpoint_path: Path,
    manifest_path: Path,
    tmp_path: Path,
    num_classes: int,
) -> None:
    onnx_p = tmp_path / "m.onnx"
    export_module.export_onnx(
        checkpoint_path,
        onnx_p,
        manifest_path=manifest_path,
        image_size=128,
        opset=17,
    )
    assert onnx_p.is_file()
    export_module.export_torchscript(
        checkpoint_path,
        tmp_path / "m.pt",
        manifest_path=manifest_path,
        image_size=128,
    )
    report = export_module.verify_onnx(
        onnx_p,
        checkpoint_path,
        manifest_path=manifest_path,
        image_size=128,
        rtol=1e-2,
        atol=1e-1,
    )
    assert report["outputs_match"] is True


def test_export_main(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
    manifest_path: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export.py",
            "--checkpoint",
            str(checkpoint_path),
            "--manifest-path",
            str(manifest_path),
            "--out-dir",
            str(tmp_path / "ex"),
            "--image-size",
            "128",
        ],
    )
    export_module.main()
    assert (tmp_path / "ex" / "model.onnx").is_file()
    assert (tmp_path / "ex" / "model_exported.pt2").is_file()
