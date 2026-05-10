"""Sanity check script for PlantDiseaseModel."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from model import build_model, model_summary_text  # noqa: E402


def main() -> None:
    """Run a forward pass and write model summary."""
    model = build_model(num_classes=38, pretrained=False, dropout=0.3)
    model.eval()

    sample = torch.randn(2, 3, 300, 300)
    with torch.inference_mode():
        logits = model(sample)

    if logits.shape != (2, 38):
        raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

    summary_path = Path(__file__).resolve().parents[1] / "model_summary.txt"
    summary_text = model_summary_text(model)
    summary_text += f"Sanity logits shape: {tuple(logits.shape)}\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Model sanity check passed. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
