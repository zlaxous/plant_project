"""Tests for evaluation metrics (no gate enforcement)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import evaluate as evaluate_module
from evaluate import _save_confusion_matrix_plot, _save_per_class_f1_plot


def test_evaluate_metrics_dict(
    checkpoint_path: Path,
    manifest_path: Path,
    num_classes: int,
) -> None:
    metrics = evaluate_module.evaluate(
        checkpoint_path,
        manifest_path,
        batch_size=2,
        num_classes=num_classes,
    )
    assert "top1_accuracy" in metrics
    assert metrics["num_test_samples"] >= 1
    assert metrics["num_classes"] == num_classes
    assert isinstance(metrics["confusion_matrix"], list)


def test_save_plots(tmp_path: Path, num_classes: int) -> None:
    cm = np.eye(num_classes)
    names = [f"c{i}" for i in range(num_classes)]
    _save_confusion_matrix_plot(cm, names, tmp_path / "cm.png")
    assert (tmp_path / "cm.png").is_file()
    report = {str(i): {"f1-score": 0.5} for i in range(num_classes)}
    _save_per_class_f1_plot(report, num_classes, tmp_path / "f1.png")
    assert (tmp_path / "f1.png").is_file()


def test_evaluate_main_skips_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    label_map_path: Path,
) -> None:
    import sys

    monkeypatch.setenv("SKIP_EVAL_GATE", "1")
    monkeypatch.setenv("PLANT_DISEASE_MANIFEST", str(manifest_path))
    monkeypatch.setenv("PLANT_DISEASE_LABEL_MAP", str(label_map_path))
    out = tmp_path / "res"
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--checkpoint", str(checkpoint_path), "--output-dir", str(out)]
    )
    evaluate_module.main()
    assert (out / "metrics.json").is_file()
    assert (out / "confusion_matrix.png").is_file()
