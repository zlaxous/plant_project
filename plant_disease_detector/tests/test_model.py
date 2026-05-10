"""Tests for model construction and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from model import (
    build_model,
    count_trainable_parameters,
    load_checkpoint_weights,
    model_summary_text,
)


def test_forward_shape(num_classes: int) -> None:
    m = build_model(num_classes=num_classes, pretrained=False)
    m.eval()
    x = torch.randn(2, 3, 300, 300)
    with torch.inference_mode():
        y = m(x)
    assert y.shape == (2, num_classes)


def test_predict_topk(num_classes: int) -> None:
    m = build_model(num_classes=num_classes, pretrained=False)
    m.eval()
    x = torch.randn(1, 3, 300, 300)
    probs, idx = m.predict_topk(x, k=2)
    assert probs.shape == (1, 2)
    assert idx.shape == (1, 2)


def test_load_checkpoint_weights(tmp_path: Path, num_classes: int) -> None:
    m = build_model(num_classes=num_classes, pretrained=False)
    ckpt = tmp_path / "c.pt"
    torch.save({"model_state": m.state_dict()}, ckpt)
    m2 = build_model(num_classes=num_classes, pretrained=False)
    meta = load_checkpoint_weights(m2, str(ckpt))
    assert "missing_keys" in meta


def test_model_summary(num_classes: int) -> None:
    m = build_model(num_classes=num_classes, pretrained=False)
    text = model_summary_text(m)
    assert "Trainable parameters" in text
    assert count_trainable_parameters(m) > 0
