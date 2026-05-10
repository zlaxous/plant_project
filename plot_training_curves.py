"""Plot Training Curves — Loss & Accuracy graphs from training history JSON.

Generates publication-quality plots showing:
1. Training Loss vs Validation Loss over epochs
2. Training Accuracy vs Validation Accuracy over epochs

Expected input format (from either scratch_cnn_history.json or train_log.json):
[
    {"epoch": 1, "train_loss": 1.23, "train_acc": 0.45, "val_loss": 1.10, "val_acc": 0.52},
    ...
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history_path: Path,
    output_path: Path,
    model_name: str = "Model",
) -> Path:
    """Load training history JSON and save loss/accuracy curves as PNG.

    Args:
        history_path: Path to JSON file with per-epoch metrics.
        output_path: Where to save the plot image.
        model_name: Label for the legend (e.g. "Scratch CNN" or "EfficientNet-B3").

    Returns:
        Path to the saved plot image.
    """
    data = json.loads(history_path.read_text(encoding="utf-8"))

    # Support both scratch_cnn_history format and train_log format
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for entry in data:
        epochs.append(entry.get("epoch", entry.get("epoch", len(epochs) + 1)))
        train_losses.append(entry.get("train_loss", entry.get("train_loss", 0)))
        train_accs.append(entry.get("train_acc", entry.get("train_acc", 0)))
        val_losses.append(entry.get("val_loss", entry.get("val_loss", 0)))
        val_accs.append(entry.get("val_acc", entry.get("val_acc", 0)))

    epochs = np.array(epochs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss Plot ──────────────────────────────────────────────────────
    ax1.plot(epochs, train_losses, "b-", linewidth=1.8, label="Training Loss", marker="o", markersize=4)
    ax1.plot(epochs, val_losses, "r-", linewidth=1.8, label="Validation Loss", marker="s", markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"{model_name} — Loss Curves", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(epochs[0], epochs[-1])

    # ── Accuracy Plot ──────────────────────────────────────────────────
    ax2.plot(epochs, train_accs, "b-", linewidth=1.8, label="Training Accuracy", marker="o", markersize=4)
    ax2.plot(epochs, val_accs, "r-", linewidth=1.8, label="Validation Accuracy", marker="s", markersize=4)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title(f"{model_name} — Accuracy Curves", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(epochs[0], epochs[-1])
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Training curves saved to: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss and accuracy curves.")
    parser.add_argument("--history", type=Path, required=True,
                        help="Path to training history JSON file")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "training_curves.png",
                        help="Output PNG path")
    parser.add_argument("--model-name", type=str, default="Model",
                        help="Model name for plot title")
    args = parser.parse_args()

    plot_training_curves(
        history_path=args.history,
        output_path=args.output,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()