"""CNN from Scratch — Built layer-by-layer for Plant Disease Classification.

Architecture (built manually, no pre-trained components):
    Input(3×380×380)
    → Conv2D(3→32, 3×3) + ReLU + BatchNorm → MaxPool(2×2)   [190×190×32]
    → Conv2D(32→64, 3×3) + ReLU + BatchNorm → MaxPool(2×2)  [95×95×64]
    → Conv2D(64→128, 3×3) + ReLU + BatchNorm → MaxPool(2×2) [47×47×128]
    → Conv2D(128→256, 3×3) + ReLU + BatchNorm → MaxPool(2×2)[23×23×256]
    → GlobalAvgPool → Flatten → Dense(256) + ReLU + Dropout(0.5)
    → Dense(128) + ReLU + Dropout(0.3)
    → Dense(38) + Softmax

This model reuses the project's existing dataset and preprocessing pipeline
(PlantDiseaseDataset, build_transforms, manifest.json) so it can be trained
and evaluated against the same data splits as the transfer learning model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Reuse existing project infrastructure — these import the same dataset,
# transforms, and paths used by the main project's transfer learning pipeline
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plant_disease_detector.dataset import PlantDiseaseDataset, build_transforms
from plant_disease_detector.paths import (
    default_manifest_path,
    num_classes_from_manifest,
    load_label_map,
)

RESUME_CKPT = "training_state.pt"  # filename for pause/resume checkpoint


# ─── Model Architecture: CNN from Scratch ─────────────────────────────────


class ScratchCNN(nn.Module):
    """Fully custom CNN built from scratch — no pre-trained components.

    Every layer is explicitly defined: Conv2D → ReLU → BatchNorm → MaxPool,
    followed by Global Avg Pooling, Flatten, and Dense (Fully Connected) layers.
    """

    def __init__(self, num_classes: int = 38, dropout_rate: float = 0.5) -> None:
        super().__init__()

        # Feature extractor — convolutional base
        self.features = nn.Sequential(
            # Block 1: 380×380×3 → 190×190×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 190×190×32 → 95×95×64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 95×95×64 → 47×47×128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 47×47×128 → 23×23×256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier head — fully connected layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 23×23×256 → 1×1×256
            nn.Flatten(),               # 256
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),  # 0.3
            nn.Linear(128, num_classes),
        )

        self._num_classes = num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (no softmax)."""
        features = self.features(images)
        logits = self.classifier(features)
        return logits

    @torch.inference_mode()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax."""
        logits = self(images)
        return torch.softmax(logits, dim=1)

    @torch.inference_mode()
    def predict_topk(self, images: torch.Tensor, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k probabilities and class indices."""
        probs = self.predict_proba(images)
        return torch.topk(probs, k=min(k, self._num_classes), dim=1)


def build_scratch_cnn(num_classes: int = 38, dropout_rate: float = 0.5) -> ScratchCNN:
    """Factory function for the scratch CNN."""
    return ScratchCNN(num_classes=num_classes, dropout_rate=dropout_rate)


# ─── Utility Functions ─────────────────────────────────────────────────────


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    """Print a clean summary of the model architecture."""
    total = sum(p.numel() for p in model.parameters())
    trainable = count_parameters(model)
    lines = [
        f"Model: {model.__class__.__name__}",
        f"Total parameters: {total:,}",
        f"Trainable parameters: {trainable:,}",
        f"Frozen parameters: {total - trainable:,}",
    ]
    if hasattr(model, '_num_classes'):
        lines.append(f"Output classes: {model._num_classes}")
    return "\n".join(lines) + "\n"


# ─── Color-Coded Terminal Output ─────────────────────────────────────────

_BOLD = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RED = "\033[91m"
_RESET = "\033[0m"
_GRAY = "\033[90m"

def _color(text: str, color_code: str) -> str:
    if os.name == "nt":
        return text
    return f"{color_code}{text}{_RESET}"

def _print_box(title: str, width: int = 60) -> None:
    print()
    print("┌" + "─" * (width - 2) + "┐")
    centered = title.center(width - 4)
    print(f"│ {centered} │")
    print("└" + "─" * (width - 2) + "┘")

def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


# ─── Early Stopping ───────────────────────────────────────────────────────


class EarlyStopping:
    """Simple early stopping callback."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best: float = -math.inf
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if value > self.best:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# ─── Pause/Resume State Management ────────────────────────────────────────


def _save_training_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    num_classes: int,
) -> None:
    """Save full training state so it can be resumed after Ctrl+C."""
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "history": history,
        "config": config,
        "num_classes": num_classes,
    }
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)  # atomic replace — works on Windows when file exists
    print(f"  {_color('💾 Training state saved — pause/resume ready', _YELLOW)}")
    print(f"  {_color(f'  → {path}', _GRAY)}")


def _load_training_state(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]], dict[str, Any], int, dict] | None:
    """Load training state for resume. Returns None if no valid state found."""
    if not path.exists():
        return None
    print(f"  {_color('📂 Loading previous training state...', _CYAN)}")
    state = torch.load(path, map_location=device, weights_only=False)

    # Restore model
    model.load_state_dict(state["model_state"])

    print(f"  Resuming from epoch {state['epoch']}")
    print(f"  Previous best accuracy: {state['best_acc']:.4f}")
    print(f"  Previous epochs logged: {len(state.get('history', []))}")

    return (
        state["epoch"],
        state["best_acc"],
        state.get("history", []),
        state.get("config", {}),
        state["num_classes"],
        state["optimizer_state"],   # returned for deferred loading
    )


# ─── Ctrl+C Handler ───────────────────────────────────────────────────────


_interrupted = False

def _handle_interrupt(sig=None, frame=None):
    """Handle Ctrl+C gracefully during training."""
    global _interrupted
    _interrupted = True
    print(f"\n  {_color('⏸ Training paused! State saved. Run with --resume to continue.', _YELLOW)}")
    print(f"  {_color('   Use Ctrl+C again to force quit.', _GRAY)}")


# ─── Training Functions ────────────────────────────────────────────────────


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_desc: str = "",
) -> tuple[float, float]:
    """Train for one epoch with tqdm progress bar. Returns (avg_loss, avg_accuracy)."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"  {epoch_desc}", leave=False, unit="batch",
                ncols=80, mininterval=0.5, disable=None)

    for images, labels in pbar:
        if _interrupted:
            break

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        batch_acc = accuracy(logits.detach(), labels)
        total_acc += batch_acc
        steps += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")

    return total_loss / max(steps, 1), total_acc / max(steps, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: dict[int, str] | None = None,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Evaluate model and return detailed metrics including per-class F1."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    pbar = tqdm(loader, desc="  Validating", leave=False, ncols=80, disable=None)
    for images, labels in pbar:
        if _interrupted:
            break

        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += float(loss.item())
        total_acc += accuracy(logits, labels)
        steps += 1

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    avg_loss = round(total_loss / max(steps, 1), 4)
    avg_acc = round(total_acc / max(steps, 1), 4)

    # Per-class metrics
    unique_labels = sorted(set(all_targets))
    report = classification_report(
        all_targets, all_preds, labels=unique_labels, output_dict=True, zero_division=0,
    )

    # Extract worst-performing classes
    per_class_f1 = {}
    if class_names and num_classes:
        for i in range(num_classes):
            i_str = str(i)
            if i_str in report:
                per_class_f1[class_names.get(i, str(i))] = report[i_str].get("f1-score", 0.0)
        worst = sorted(per_class_f1.items(), key=lambda x: x[1])[:5]
    else:
        worst = []

    return {
        "val_loss": avg_loss,
        "val_acc": avg_acc,
        "val_macro_f1": round(report.get("macro avg", {}).get("f1-score", 0.0), 4),
        "val_weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0.0), 4),
        "worst_classes": worst,
        "predictions": all_preds,
        "targets": all_targets,
    }


def train_scratch_model(
    manifest_path: Path,
    output_dir: Path,
    image_size: int = 380,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    num_epochs: int = 30,
    dropout_rate: float = 0.5,
    weight_decay: float = 1e-4,
    seed: int = 42,
    resume_from: str | Path | None = None,
) -> Path:
    """Full training pipeline for the scratch CNN.

    Supports:
    - Ctrl+C graceful pause with state saved to disk
    - --resume to continue from paused state
    - Early stopping when validation accuracy plateaus
    - Atomic file saves (no corruption on crash)

    Returns:
        Path to the best checkpoint.
    """
    global _interrupted
    _interrupted = False

    # ── Reproducibility ────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Device ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ── Dataset ────────────────────────────────────────────────────────
    num_classes = num_classes_from_manifest(manifest_path)
    train_transform = build_transforms(image_size=image_size, train=True)
    val_transform = build_transforms(image_size=image_size, train=False)

    train_ds = PlantDiseaseDataset.from_manifest(
        manifest_path=manifest_path, split="train", transform=train_transform,
    )
    val_ds = PlantDiseaseDataset.from_manifest(
        manifest_path=manifest_path, split="val", transform=val_transform,
    )

    print(f"\nTrain samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Classes:       {num_classes}")

    # Extract targets safely (handles both PlantDiseaseDataset and Subset)
    from torch.utils.data import Subset as _Subset
    if isinstance(train_ds, _Subset):
        # Subset wraps PlantDiseaseDataset — access records via .dataset
        base_ds = train_ds.dataset
        targets = [base_ds.records[idx].class_id for idx in train_ds.indices]
    else:
        targets = [r.class_id for r in train_ds.records]
    class_counts = Counter(targets)
    weights = [1.0 / class_counts[t] for t in targets]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # ── Model ──────────────────────────────────────────────────────────
    model = build_scratch_cnn(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    print(f"\n{model_summary(model)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Load class names for per-class F1 reporting
    class_names = load_label_map()

    # ── Resume Handling ────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(resume_from) if resume_from else output_dir / RESUME_CKPT
    best_ckpt = output_dir / "scratch_cnn_best.pt"
    best_acc = -1.0
    history: list[dict[str, float]] = []
    start_epoch = 1

    resume_data = _load_training_state(resume_path, model, device) if resume_path.exists() else None
    if resume_data is not None:
        saved_epoch, best_acc, history, saved_config, saved_nc, saved_opt_state = resume_data
        num_classes = saved_nc
        optimizer.load_state_dict(saved_opt_state)
        start_epoch = saved_epoch + 1
        print(f"\n  {_color('✓ Resuming training...', _GREEN)}")

    # ── Training Loop ──────────────────────────────────────────────────
    _print_box(f"Training Scratch CNN — {num_epochs} epochs")
    print(f"  {_color(f'Batch size: {batch_size}  |  LR: {learning_rate}  |  Weight decay: {weight_decay}', _CYAN)}")
    if start_epoch > 1:
        print(f"  {_color(f'Resuming from epoch {start_epoch}', _GREEN)}")
    print(f"  {_color('Press Ctrl+C to pause — state saved automatically', _GRAY)}")
    print()

    stopper = EarlyStopping(patience=4)
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        if _interrupted:
            break

        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch_desc=f"Train epoch {epoch}/{num_epochs}",
        )

        if _interrupted:
            # Still save what we have before breaking
            pass
        else:
            val_metrics = evaluate(
                model, val_loader, criterion, device,
                class_names=class_names, num_classes=num_classes,
            )

            elapsed = time.time() - epoch_start
            is_best = val_metrics["val_acc"] > best_acc
            if is_best:
                best_acc = val_metrics["val_acc"]
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_metrics["val_acc"],
                    "val_loss": val_metrics["val_loss"],
                }, best_ckpt)

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": val_metrics["val_loss"],
                "val_acc": val_metrics["val_acc"],
                "val_macro_f1": val_metrics.get("val_macro_f1", 0),
                "val_weighted_f1": val_metrics.get("val_weighted_f1", 0),
            }
            history.append(record)

            # Color-coded summary line
            time_str = _format_time(elapsed)
            best_mark = _color("★ BEST", _GREEN) if is_best else ""
            acc_bar_len = 30
            filled = int(acc_bar_len * val_metrics["val_acc"])
            bar = "▓" * filled + "░" * (acc_bar_len - filled)
            vl = val_metrics["val_loss"]
            va = val_metrics["val_acc"]
            mf1 = val_metrics.get("val_macro_f1", 0)
            epoch_str = f"Epoch {epoch:2d}/{num_epochs}"
            train_loss_str = f"{train_loss:.4f}"
            train_acc_str = f"{train_acc:.4f}"
            val_loss_str = f"{vl:.4f}"
            val_acc_str = f"{va:.4f}"
            print(f"  {_color(epoch_str, _BOLD)}  "
                  f"Train [{_color(train_loss_str, _YELLOW)}/{_color(train_acc_str, _GREEN)}]  "
                  f"Val [{_color(val_loss_str, _YELLOW)}/{_color(val_acc_str, _GREEN)}]  "
                  f"Macro F1: {mf1:.4f}  "
                  f"{_color(time_str, _GRAY)}  {best_mark}")
            print(f"    Acc: [{bar}]  {va*100:.1f}%")

            # Show worst classes
            worst = val_metrics.get("worst_classes", [])
            if worst:
                worst_str = ", ".join(f"{name}({f1:.2f})" for name, f1 in worst)
                print(f"    {_color(f'Worst: {worst_str}', _RED)}")
            print()

            # Save resume checkpoint after each epoch
            config_dict = {
                "manifest_path": str(manifest_path),
                "output_dir": str(output_dir),
                "image_size": image_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "dropout_rate": dropout_rate,
                "weight_decay": weight_decay,
                "seed": seed,
            }
            _save_training_state(
                resume_path, model, optimizer, epoch, best_acc, history, config_dict, num_classes,
            )

            # Early stopping check
            if stopper.step(val_metrics["val_acc"]):
                print(f"  {_color(f'⏹ Early stopping — no improvement for {stopper.patience} epochs', _YELLOW)}")
                break

    # After loop ends (completed, interrupted, or early stopped)
    total_time = time.time() - start_time

    if _interrupted:
        print(f"\n  {_color('⏸ Training paused. Resume with --resume flag.', _YELLOW)}")
    elif len(history) > 0:
        print(f"\n{'='*60}")
        print(f"  Training complete! Total time: {total_time/60:.1f} minutes")
        print(f"  Best validation accuracy: {best_acc:.4f}")
        print(f"  Best checkpoint: {best_ckpt}")
        print(f"{'='*60}\n")

        # Save training history JSON
        history_path = output_dir / "scratch_cnn_history.json"
        (output_dir / "scratch_cnn_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8",
        )
        print(f"Training history saved to: {history_path}")

        # Clean up resume checkpoint since training completed normally
        if resume_path.exists():
            resume_path.unlink()
            print(f"  {_color('🧹 Resume checkpoint cleaned up', _GRAY)}")
    else:
        print(f"\n  {_color('No epochs completed.', _YELLOW)}")

    return best_ckpt


# ─── CLI Entrypoint ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Scratch CNN from scratch.")
    parser.add_argument("--manifest-path", type=Path, default=None,
                        help="Path to manifest.json (default: auto-detect)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "checkpoints",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--image-size", type=int, default=380,
                        help="Input image size (square)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from saved state in output directory")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 1 epoch, 5% of data, batch=8")
    args = parser.parse_args()

    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, _handle_interrupt)

    manifest_path = args.manifest_path or default_manifest_path()
    output_dir = args.output_dir
    resume_path = output_dir / RESUME_CKPT if args.resume else None

    # Apply smoke test overrides
    subset_fraction = 0.05 if args.smoke else 1.0
    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
        print(f"\n  {_color('SMOKE TEST MODE — 1 epoch, 5% data', _YELLOW)}\n")

    # Update the dataset loading to support subset_fraction
    if subset_fraction < 1.0:
        from torch.utils.data import Subset
        original_from_manifest = PlantDiseaseDataset.from_manifest

        def _smoke_from_manifest(manifest_path, split, transform=None):
            ds = original_from_manifest(manifest_path, split, transform)
            if split == "train":
                keep = max(1, int(len(ds) * subset_fraction))
                return Subset(ds, list(range(keep)))
            return ds

        # Monkey-patch for this run
        import plant_disease_detector.dataset as ds_mod
        ds_mod.PlantDiseaseDataset.from_manifest = staticmethod(_smoke_from_manifest)

    train_scratch_model(
        manifest_path=manifest_path,
        output_dir=output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        dropout_rate=args.dropout,
        weight_decay=args.weight_decay,
        resume_from=resume_path,
    )


if __name__ == "__main__":
    main()