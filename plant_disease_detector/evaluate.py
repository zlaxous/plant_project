"""Evaluation script with metrics, plots, and optional top-1 gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import PlantDiseaseDataset, build_transforms
from model import build_model, load_checkpoint_weights
from paths import (
    default_manifest_path,
    default_results_dir,
    load_label_map,
    num_classes_from_manifest,
)

GATE_THRESHOLD = 0.95


def _gate_skipped() -> bool:
    skip = os.environ.get("SKIP_EVAL_GATE", "").lower() in ("1", "true", "yes")
    ci = os.environ.get("CI", "").lower() in ("1", "true", "yes")
    return skip or ci


def evaluate(
    checkpoint_path: Path,
    manifest_path: Path,
    batch_size: int = 32,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Evaluate model on test split and return metrics."""
    if num_classes is None:
        num_classes = num_classes_from_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    load_meta = load_checkpoint_weights(model, str(checkpoint_path))
    model.eval()

    test_ds = PlantDiseaseDataset.from_manifest(
        manifest_path=manifest_path,
        split="test",
        transform=build_transforms(image_size=380, train=False),
    )
    if len(test_ds) == 0:
        raise RuntimeError("Test split is empty. Run prepare_data.py and train.py first.")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    preds: list[int] = []
    targets: list[int] = []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    labels_list = list(range(num_classes))
    top1 = sum(int(p == t) for p, t in zip(preds, targets, strict=False)) / max(len(targets), 1)
    cls_report = classification_report(
        targets, preds, labels=labels_list, output_dict=True, zero_division=0
    )
    conf_matrix = confusion_matrix(targets, preds, labels=labels_list)
    gate_passed = top1 >= GATE_THRESHOLD

    return {
        "top1_accuracy": top1,
        "gate_threshold": GATE_THRESHOLD,
        "gate_passed": gate_passed,
        "num_test_samples": len(targets),
        "num_classes": num_classes,
        "classification_report": cls_report,
        "confusion_matrix": conf_matrix.tolist(),
        "checkpoint_meta": load_meta,
    }


def _save_confusion_matrix_plot(matrix: np.ndarray, class_names: list[str], out_path: Path) -> None:
    """Save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_per_class_f1_plot(cls_report: dict[str, Any], num_classes: int, out_path: Path) -> None:
    """Bar chart of per-class F1 scores."""
    f1_scores: list[float] = []
    for i in range(num_classes):
        key = str(i)
        if key in cls_report and isinstance(cls_report[key], dict):
            f1_scores.append(float(cls_report[key].get("f1-score", 0.0)))
        else:
            f1_scores.append(0.0)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(num_classes)
    ax.bar(x, f1_scores, color="steelblue")
    ax.set_xlabel("Class id")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 (test set)")
    ax.set_xticks(x)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Entrypoint for evaluation, plots, and gate enforcement."""
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on test split.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest_path = args.manifest_path or default_manifest_path()
    output_dir = args.output_dir or default_results_dir()
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parent / "checkpoints" / "best.pt"
    num_classes = num_classes_from_manifest(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(
        checkpoint, manifest_path, batch_size=args.batch_size, num_classes=num_classes
    )

    label_map = load_label_map()
    class_names = [label_map.get(i, str(i)) for i in range(num_classes)]

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.4f} (gate {GATE_THRESHOLD})")

    cm = np.array(metrics["confusion_matrix"], dtype=np.float64)
    _save_confusion_matrix_plot(cm, class_names, output_dir / "confusion_matrix.png")
    _save_per_class_f1_plot(
        metrics["classification_report"], num_classes, output_dir / "per_class_f1.png"
    )
    print(f"Saved plots under {output_dir}")

    if not metrics["gate_passed"]:
        msg = (
            f"Evaluation gate not met: top1={metrics['top1_accuracy']:.4f} < {GATE_THRESHOLD}. "
            "Train longer on the full PlantVillage split; tune augmentations and learning rate."
        )
        metrics["blocked_reason"] = msg
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if _gate_skipped():
            print(f"SKIP_EVAL_GATE/CI set — continuing without failing: {msg}")
            return
        raise SystemExit(f"BLOCKED: evaluation_engineer | {msg}")

    print("Gate passed: top1 >= 0.95")


if __name__ == "__main__":
    main()
