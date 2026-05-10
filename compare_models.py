"""Model Comparison — Generates a comparison table between Scratch CNN and Transfer Learning.

This script:
1. Loads both trained models (Scratch CNN from scratch_cnn_best.pt, Transfer Learning from best.pt)
2. Runs both on the same test split
3. Computes and saves a comparison table with:
   - Training Accuracy vs Validation Accuracy
   - Training Time (manual — read from training logs)
   - Number of Trainable Parameters
   - Final Test Set Performance (Accuracy, Precision, Recall, F1)
   - Inference Time per sample

Usage:
    python compare_models.py \\
        --scratch-checkpoint disscution_project/checkpoints/scratch_cnn_best.pt \\
        --transfer-checkpoint plant_disease_detector/checkpoints/best.pt \\
        --manifest-path plant_disease_detector/data/manifest.json \\
        --scratch-history disscution_project/checkpoints/scratch_cnn_history.json \\
        --transfer-history plant_disease_detector/checkpoints/logs/train_log.json \\
        --output-dir disscution_project/results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import existing project infrastructure
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plant_disease_detector.dataset import PlantDiseaseDataset, build_transforms
from plant_disease_detector.paths import (
    default_manifest_path,
    num_classes_from_manifest,
    load_label_map,
)
from plant_disease_detector.model import build_model, load_checkpoint_weights, count_trainable_parameters as count_transfer_params

# Import scratch CNN
from CNN_from_scratch import build_scratch_cnn, count_parameters as count_scratch_params


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    """Run full evaluation on a model. Returns accuracy + classification report + inference timing."""
    model.eval()

    all_preds: list[int] = []
    all_targets: list[int] = []
    inference_times: list[float] = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start = time.perf_counter()
            logits = model(images)
            elapsed = time.perf_counter() - start
            inference_times.append(elapsed / images.size(0))  # per-sample

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    accuracy = sum(int(p == t) for p, t in zip(all_preds, all_targets)) / max(len(all_targets), 1)
    labels_list = list(range(num_classes))
    report = classification_report(
        all_targets, all_preds, labels=labels_list, output_dict=True, zero_division=0,
    )

    return {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "predictions": all_preds,
        "targets": all_targets,
        "inference_time_per_sample_ms": round(
            float(np.mean(inference_times)) * 1000, 2
        ),
        "inference_time_std_ms": round(
            float(np.std(inference_times)) * 1000, 2
        ),
    }


def load_training_stats(history_path: Path | None) -> dict[str, Any]:
    """Extract best accuracy and training time from a history JSON."""
    if history_path is None or not history_path.exists():
        return {"best_val_acc": "N/A", "epochs_trained": "N/A"}

    data = json.loads(history_path.read_text(encoding="utf-8"))
    best_entry = max(data, key=lambda x: x.get("val_acc", 0))
    return {
        "best_val_acc": round(best_entry.get("val_acc", 0), 4),
        "epochs_trained": best_entry.get("epoch", len(data)),
    }


def generate_comparison_table(
    scratch_metrics: dict[str, Any],
    transfer_metrics: dict[str, Any],
    scratch_params: int,
    transfer_params: int,
    scratch_stats: dict[str, Any],
    transfer_stats: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Generate comparison table as JSON and Markdown."""
    comparison = {
        "model": ["Scratch CNN", "Transfer Learning (EfficientNet-B3)"],
        "architecture": [
            "4 Conv Blocks → GlobalAvgPool → 2 Dense layers (from scratch)",
            "EfficientNet-B3 → Linear(1536→512) → BatchNorm → Dropout → Linear(38)",
        ],
        "trainable_parameters": [scratch_params, transfer_params],
        "total_parameters": [scratch_params, transfer_params],
        "best_val_accuracy": [scratch_stats.get("best_val_acc", "N/A"), transfer_stats.get("best_val_acc", "N/A")],
        "epochs_trained": [scratch_stats.get("epochs_trained", "N/A"), transfer_stats.get("epochs_trained", "N/A")],
        "test_accuracy": [
            scratch_metrics.get("accuracy", "N/A"),
            transfer_metrics.get("accuracy", "N/A"),
        ],
        "test_precision_macro": [
            round(scratch_metrics["classification_report"].get("macro avg", {}).get("precision", 0), 4),
            round(transfer_metrics["classification_report"].get("macro avg", {}).get("precision", 0), 4),
        ],
        "test_recall_macro": [
            round(scratch_metrics["classification_report"].get("macro avg", {}).get("recall", 0), 4),
            round(transfer_metrics["classification_report"].get("macro avg", {}).get("recall", 0), 4),
        ],
        "test_f1_macro": [
            round(scratch_metrics["classification_report"].get("macro avg", {}).get("f1-score", 0), 4),
            round(transfer_metrics["classification_report"].get("macro avg", {}).get("f1-score", 0), 4),
        ],
        "test_f1_weighted": [
            round(scratch_metrics["classification_report"].get("weighted avg", {}).get("f1-score", 0), 4),
            round(transfer_metrics["classification_report"].get("weighted avg", {}).get("f1-score", 0), 4),
        ],
        "inference_time_per_sample_ms": [
            scratch_metrics.get("inference_time_per_sample_ms", "N/A"),
            transfer_metrics.get("inference_time_per_sample_ms", "N/A"),
        ],
    }

    # Save as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "comparison_table.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8",
    )

    # Save as Markdown table
    md_lines = [
        "# Model Comparison Table\n",
        "| Metric | Scratch CNN | Transfer Learning (EfficientNet-B3) |",
        "|--------|-------------|-------------------------------------|",
    ]
    rows = [
        ("Architecture", comparison["architecture"]),
        ("Trainable Parameters", f"{comparison['trainable_parameters'][0]:,}", f"{comparison['trainable_parameters'][1]:,}"),
        ("Total Parameters", f"{comparison['total_parameters'][0]:,}", f"{comparison['total_parameters'][1]:,}"),
        ("Best Validation Accuracy", str(comparison["best_val_accuracy"][0]), str(comparison["best_val_accuracy"][1])),
        ("Epochs Trained", str(comparison["epochs_trained"][0]), str(comparison["epochs_trained"][1])),
        ("Test Accuracy", str(comparison["test_accuracy"][0]), str(comparison["test_accuracy"][1])),
        ("Precision (Macro)", str(comparison["test_precision_macro"][0]), str(comparison["test_precision_macro"][1])),
        ("Recall (Macro)", str(comparison["test_recall_macro"][0]), str(comparison["test_recall_macro"][1])),
        ("F1-Score (Macro)", str(comparison["test_f1_macro"][0]), str(comparison["test_f1_macro"][1])),
        ("F1-Score (Weighted)", str(comparison["test_f1_weighted"][0]), str(comparison["test_f1_weighted"][1])),
        ("Inference Time / Sample (ms)", str(comparison["inference_time_per_sample_ms"][0]), str(comparison["inference_time_per_sample_ms"][1])),
    ]
    for row in rows:
        md_lines.append(f"| {' | '.join(str(c) for c in row)} |")

    md_content = "\n".join(md_lines) + "\n"
    output_path.write_text(md_content, encoding="utf-8")
    print(f"Comparison table saved: {output_path}")

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Scratch CNN vs Transfer Learning.")
    parser.add_argument("--scratch-checkpoint", type=Path, required=True,
                        help="Path to scratch CNN checkpoint (scratch_cnn_best.pt)")
    parser.add_argument("--transfer-checkpoint", type=Path, required=True,
                        help="Path to transfer learning checkpoint (best.pt)")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--scratch-history", type=Path, default=None,
                        help="Scratch CNN training history JSON")
    parser.add_argument("--transfer-history", type=Path, default=None,
                        help="Transfer learning training history JSON")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=380)
    args = parser.parse_args()

    manifest_path = args.manifest_path or default_manifest_path()
    num_classes = num_classes_from_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ────────────────────────────────────────────────────────
    test_transform = build_transforms(image_size=args.image_size, train=False)
    test_ds = PlantDiseaseDataset.from_manifest(
        manifest_path=manifest_path, split="test", transform=test_transform,
    )
    if len(test_ds) == 0:
        raise RuntimeError("Test split is empty. Use build_manifest.py first.")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Test samples: {len(test_ds)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Scratch CNN ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Loading Scratch CNN...")
    scratch_model = build_scratch_cnn(num_classes=num_classes).to(device)
    scratch_checkpoint = torch.load(args.scratch_checkpoint, map_location=device, weights_only=False)
    scratch_model.load_state_dict(scratch_checkpoint["model_state"])
    scratch_params = count_scratch_params(scratch_model)
    print(f"  Parameters: {scratch_params:,}")

    scratch_metrics = evaluate_model(scratch_model, test_loader, device, num_classes)
    print(f"  Test Accuracy: {scratch_metrics['accuracy']:.4f}")
    print(f"  Inference: {scratch_metrics['inference_time_per_sample_ms']:.2f} ms/sample")

    # ── Load Transfer Learning ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  Loading Transfer Learning (EfficientNet-B3)...")
    transfer_model = build_model(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint_weights(transfer_model, str(args.transfer_checkpoint))
    transfer_params = count_transfer_params(transfer_model)
    print(f"  Parameters: {transfer_params:,}")

    transfer_metrics = evaluate_model(transfer_model, test_loader, device, num_classes)
    print(f"  Test Accuracy: {transfer_metrics['accuracy']:.4f}")
    print(f"  Inference: {transfer_metrics['inference_time_per_sample_ms']:.2f} ms/sample")

    # ── Training Stats ─────────────────────────────────────────────────
    scratch_stats = load_training_stats(args.scratch_history)
    transfer_stats = load_training_stats(args.transfer_history)

    # ── Comparison Table ───────────────────────────────────────────────
    md_path = args.output_dir / "comparison_table.md"
    generate_comparison_table(
        scratch_metrics, transfer_metrics,
        scratch_params, transfer_params,
        scratch_stats, transfer_stats,
        md_path,
    )

    print(f"\n{'='*60}")
    print("  Comparison complete! Results saved to:")
    print(f"    {args.output_dir / 'comparison_table.md'}")
    print(f"    {args.output_dir / 'comparison_table.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()