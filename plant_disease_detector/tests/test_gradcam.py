"""Lightweight Grad-CAM tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gradcam import _overlay_heatmap, generate_gradcam, generate_gradcam_grid


def test_overlay_heatmap_shape() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.float32) * 0.5
    out = _overlay_heatmap(rgb, mask)
    assert out.shape == (32, 32, 3)


def test_generate_gradcam_grid(
    tiny_image: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    tmp_path: Path,
) -> None:
    import shutil

    out = tmp_path / "grid.png"
    img2 = tmp_path / "b.jpg"
    shutil.copy(tiny_image, img2)
    generate_gradcam_grid(
        [tiny_image, img2], checkpoint_path, out, manifest_path=manifest_path, ncols=2
    )
    assert out.is_file()


def test_generate_gradcam_single_file(
    tiny_image: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "one.png"
    generate_gradcam(tiny_image, checkpoint_path, out, manifest_path=manifest_path)
    assert out.is_file()
