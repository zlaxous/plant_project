"""ROC/AUC Curves — Multi-class One-vs-Rest evaluation.

Generates ROC curves for both models (Scratch CNN and Transfer Learning)
on the test set, with AUC scores per class.

Bonus requirement from Section 5 of the project guidelines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plant_disease_detector.dataset import PlantDiseaseDataset, build_transforms
from plant_disease_detector.paths import (
    default_manifest_path,
    num_classes_from_manifest,
    load_label_map,
)
from plant_disease_detector.model import build_model, load_checkpoint_weights
from CNN_from_scratch import build_scratch_cnn


def plot_roc_curves(
    model: torch.nn.Module,
    test_loader: DataLoader,
    num_classes: int,
    class_names: dict[int, str],
    model_name: str,
    output_path: Path,
    max_classes_plot: int = 10,
) -> dict:
    """Generate ROC curves and return per-class AUC scores.

    Args:
        model: Trained PyTorch model in eval mode.
        test_loader: DataLoader for test split.
        num_classes: Number of classes.
        class_names: Dict mapping class_id → class name.
        model_name: Label for plot title.
        output_path: Where to save the PNG.
        max_classes_plot: Max classes to show in the legend.

    Returns:
        Dict of per-class AUC scores.
    """
    device = next(model.parameters()).device
    model.eval()

    # Collect all predictions and targets
    y_true: list[int] = []
    y_score: list[np.ndarray] = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Binarize labels for One-vs-Rest ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # Compute ROC and AUC for each class
    fpr: dict[int, np.ndarray] = {}
    tpr: dict[int, np.ndarray] = {}
    roc_auc: dict[int, float] = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = round(auc(fpr[i], tpr[i]), 4)

    # Sort classes by AUC (best first for display)
    sorted_classes = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.5)")

    # Plot each class (limit to max_classes_plot for readability)
    colors = plt.cm.tab10(np.linspace(0, 1, min(max_classes_plot, num_classes)))
    for idx, (class_id, auc_score) in enumerate(sorted_classes[:max_classes_plot]):
        label = class_names.get(class_id, str(class_id)).replace("___", " - ").replace("_", " ")[:30]
        ax.plot(
            fpr[class_id], tpr[class_id],
            color=colors[idx], linewidth=1.5,
            label=f"{label} (AUC={auc_score:.3f})",
        )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model_name} (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)

    # Add mean AUC text box
    mean_auc = float(np.mean(list(roc_auc.values())))
    ax.text(
        0.02, 0.98, f"Mean AUC: {mean_auc:.4f}",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"ROC curve saved: {output_path}")
    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"Top-5 classes by AUC:")
    for class_id, auc_score in sorted_classes[:5]:
        name = class_names.get(class_id, str(class_id)).replace("___", " - ").replace("_", " ")[:40]
        print(f"  {name}: {auc_score:.4f}")

    return {
        "model_name": model_name,
        "mean_auc": mean_auc,
        "per_class_auc": {str(k): v for k, v in roc_auc.items()},
        "num_classes_plotted": min(max_classes_plot, num_classes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ROC curves for both models.")
    parser.add_argument("--scratch-checkpoint", type=Path, required=True,
                        help="Path to scratch CNN checkpoint")
    parser.add_argument("--transfer-checkpoint", type=Path, required=True,
                        help="Path to transfer learning checkpoint")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=380)
    args = parser.parse_args()

    manifest_path = args.manifest_path or default_manifest_path()
    num_classes = num_classes_from_manifest(manifest_path)
    label_map = load_label_map()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test dataset
    test_transform = build_transforms(image_size=args.image_size, train=False)
    test_ds = PlantDiseaseDataset.from_manifest(
        manifest_path=manifest_path, split="test", transform=test_transform,
    )
    if len(test_ds) == 0:
        raise RuntimeError("Test split is empty.")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Test samples: {len(test_ds)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── Scratch CNN ────────────────────────────────────────────────
    if args.scratch_checkpoint.exists():
        print("\n--- Scratch CNN ---")
        scratch_model = build_scratch_cnn(num_classes=num_classes).to(device)
        ckpt = torch.load(args.scratch_checkpoint, map_location=device, weights_only=False)
        scratch_model.load_state_dict(ckpt["model_state"])
        scratch_model.eval()
        scratch_results = plot_roc_curves(
            scratch_model, test_loader, num_classes, label_map,
            "Scratch CNN",
            args.output_dir / "roc_scratch_cnn.png",
        )
        all_results["scratch_cnn"] = scratch_results

    # ── Transfer Learning ─────────────────────────────────────────
    if args.transfer_checkpoint.exists():
        print("\n--- Transfer Learning (EfficientNet-B3) ---")
        transfer_model = build_model(num_classes=num_classes, pretrained=False).to(device)
        load_checkpoint_weights(transfer_model, str(args.transfer_checkpoint))
        transfer_model.eval()
        transfer_results = plot_roc_curves(
            transfer_model, test_loader, num_classes, label_map,
            "EfficientNet-B3 (Transfer Learning)",
            args.output_dir / "roc_transfer_learning.png",
        )
        all_results["transfer_learning"] = transfer_results

    # Save summary JSON
    summary_path = args.output_dir / "roc_auc_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")

    # Comparison
    if "scratch_cnn" in all_results and "transfer_learning" in all_results:
        s = all_results["scratch_cnn"]["mean_auc"]
        t = all_results["transfer_learning"]["mean_auc"]
        print(f"\n=== AUC Comparison ===")
        print(f"  Scratch CNN:           {s:.4f}")
        print(f"  Transfer Learning:     {t:.4f}")
        print(f"  Difference:            {abs(s - t):.4f}")
        print(f"  Winner:                {'Transfer' if t > s else 'Scratch'}")


if __name__ == "__main__":
    main()