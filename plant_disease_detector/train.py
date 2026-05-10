"""Training pipeline for PlantDiseaseModel with detailed CLI monitoring."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suppress HuggingFace Hub warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from dataset import PlantDiseaseDataset, build_transforms
from model import build_model, count_trainable_parameters
from paths import num_classes_from_manifest, load_label_map

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration values."""

    manifest_path: Path
    output_dir: Path
    image_size: int = 380
    batch_size: int = 32
    num_workers: int = 0
    warmup_epochs: int = 3
    finetune_epochs: int = 12
    lr_head: float = 5e-4
    lr_finetune: float = 1.5e-4
    weight_decay: float = 1e-5
    patience: int = 4
    label_smoothing: float = 0.05
    amp: bool = True
    subset_fraction: float = 1.0
    seed: int = 42


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


# ─── CLI Logging Utilities ───────────────────────────────────────────────────

_BOLD = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RED = "\033[91m"
_RESET = "\033[0m"
_GRAY = "\033[90m"

def _color(text: str, color_code: str) -> str:
    if os.name == "nt":  # Windows
        return text
    return f"{color_code}{text}{_RESET}"


def _print_box(title: str, width: int = 72) -> None:
    print()
    print("┌" + "─" * (width - 2) + "┐")
    centered = title.center(width - 4)
    print(f"│ {centered} │")
    print("└" + "─" * (width - 2) + "┘")


def _print_header(text: str, width: int = 72, char: str = "─") -> None:
    print(f"\n  {_color(text, _CYAN)}")
    print(f"  {char * (width - 2)}")


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


# ─── Core Training Functions ─────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subset_if_needed(
    dataset: PlantDiseaseDataset, fraction: float
) -> PlantDiseaseDataset | Subset[Any]:
    if fraction >= 1.0:
        return dataset
    keep = max(1, int(len(dataset) * fraction))
    return Subset(dataset, list(range(keep)))


def _extract_targets(dataset: PlantDiseaseDataset | Subset[Any]) -> list[int]:
    if isinstance(dataset, PlantDiseaseDataset):
        return [record.class_id for record in dataset.records]
    base = dataset.dataset
    if not isinstance(base, PlantDiseaseDataset):
        raise TypeError("Subset must wrap PlantDiseaseDataset")
    return [base.records[idx].class_id for idx in dataset.indices]


def _build_weighted_sampler(dataset: PlantDiseaseDataset | Subset[Any]) -> WeightedRandomSampler:
    targets = _extract_targets(dataset)
    class_counts = Counter(targets)
    weights = [1.0 / class_counts[target] for target in targets]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    desc: str = "",
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"  {desc}", leave=False, unit="batch",
                ncols=80, mininterval=0.5, disable=None)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_train and optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        batch_acc = _accuracy(logits.detach(), labels)
        total_acc += batch_acc
        steps += 1

        if is_train:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")

    return total_loss / max(steps, 1), total_acc / max(steps, 1)


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    class_names: dict[int, str] | None = None,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Evaluate model and return detailed metrics."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="  Validating", leave=False, ncols=80, disable=None):
            images = images.to(device)
            labels = labels.to(device)
            with autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += float(loss.item())
            total_acc += _accuracy(logits, labels)
            steps += 1
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    unique_labels = sorted(set(all_targets))
    report = classification_report(
        all_targets, all_preds, labels=unique_labels, output_dict=True, zero_division=0
    )

    # Per-class F1 and worst classes
    per_class_f1 = {}
    if class_names and num_classes:
        for i in range(num_classes):
            i_str = str(i)
            if i_str in report:
                per_class_f1[class_names.get(i, str(i))] = report[i_str].get("f1-score", 0.0)
        worst = sorted(per_class_f1.items(), key=lambda x: x[1])[:5]
    else:
        worst = []

    avg_loss = total_loss / max(steps, 1)
    avg_acc = total_acc / max(steps, 1)

    return {
        "val_loss": round(avg_loss, 4),
        "val_acc": round(avg_acc, 4),
        "val_macro_f1": round(report.get("macro avg", {}).get("f1-score", 0.0), 4),
        "val_macro_recall": round(report.get("macro avg", {}).get("recall", 0.0), 4),
        "val_weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0.0), 4),
        "worst_classes": worst,
    }


def _print_epoch_summary(
    phase: str,
    epoch: int,
    total_epochs: int,
    lr: float,
    stats: dict[str, Any],
    elapsed: float,
    model: nn.Module,
    is_best: bool,
    device: torch.device,
) -> None:
    """Print a clean, informative per-epoch summary."""
    trainable = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())

    print()
    print(f"  {_color(f'═ Epoch {epoch}/{total_epochs} ─ {phase.upper():8s} ─ LR: {lr:.2e} ─ Trainable: {trainable:,}/{total_params:,} params ─ Device: {device}', _BOLD)}")
    print(f"  ─────────────────────────────────────────────────────────────────────")
    print(f"    Train → loss: {stats['train_loss']:.4f}   acc: {stats['train_acc']:.4f}")
    print(f"    Valid → loss: {stats['val_loss']:.4f}   acc: {stats['val_acc']:.4f}  ", end="")
    print(f"  Macro F1: {stats['val_macro_f1']:.4f}  Wgt F1: {stats['val_weighted_f1']:.4f}")

    # Progress bar showing accuracy
    acc_pct = stats['val_acc'] * 100
    bar_len = 40
    filled = int(bar_len * stats['val_acc'])
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"    Acc:  [{bar}] {acc_pct:.1f}%")

    # Time and status
    time_str = _format_time(elapsed)
    best_mark = _color("★ BEST (saved)", _GREEN) if is_best else ""
    early_mark = _color("(early stopping candidate)", _YELLOW) if elapsed else ""
    print(f"    Time: {time_str:>8s}   {best_mark} {early_mark}")

    # Worst classes
    worst = stats.get("worst_classes", [])
    if worst:
        worst_str = ", ".join(f"{name}({f1:.2f})" for name, f1 in worst)
        print(f"    Worst: {worst_str}")

    print()


def _export_for_deployment(
    model: nn.Module, checkpoint_path: Path, output_dir: Path, device: torch.device
) -> None:
    """Export trained model for optimized inference deployment."""
    try:
        model.eval()
        exports_dir = output_dir.parent / "exports"
        exports_dir.mkdir(exist_ok=True)

        dummy_input = torch.randn(1, 3, 380, 380, device=device)

        # TorchScript
        traced_model = torch.jit.trace(model, dummy_input)
        ts_path = exports_dir / "model_torchscript.pt"
        traced_model.save(str(ts_path))
        ts_size = ts_path.stat().st_size / 1e6
        print(f"  ✓ TorchScript exported → {ts_path} ({ts_size:.1f} MB)")

        # ONNX
        onnx_path = exports_dir / "model.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=["image"], output_names=["logits"],
            opset_version=14, verbose=False,
        )
        onnx_size = onnx_path.stat().st_size / 1e6
        print(f"  ✓ ONNX exported → {onnx_path} ({onnx_size:.1f} MB)")

    except Exception as e:
        print(f"  Warning: Could not export model: {e}")


RESUME_CKPT = "training_state.pt"  # filename for pause/resume checkpoint


def _save_training_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    phase: str,
    phase_epoch: int,  # epoch within current phase
    best_acc: float,
    log_records: list[dict[str, Any]],
    config: TrainConfig,
    num_classes: int,
) -> None:
    """Save full training state so it can be resumed after Ctrl+C."""
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "phase": phase,
        "phase_epoch": phase_epoch,
        "best_acc": best_acc,
        "log_records": log_records,
        "config": {
            "manifest_path": str(config.manifest_path),
            "output_dir": str(config.output_dir),
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "warmup_epochs": config.warmup_epochs,
            "finetune_epochs": config.finetune_epochs,
            "lr_head": config.lr_head,
            "lr_finetune": config.lr_finetune,
            "weight_decay": config.weight_decay,
            "patience": config.patience,
            "label_smoothing": config.label_smoothing,
            "amp": config.amp,
            "subset_fraction": config.subset_fraction,
            "seed": config.seed,
        },
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
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, str, int, float, list[dict[str, Any]], TrainConfig, int, dict, dict] | None:
    """Load training state for resume. Returns None if no valid state found."""
    if not path.exists():
        return None
    print(f"  {_color('📂 Loading previous training state...', _CYAN)}")
    state = torch.load(path, map_location=device, weights_only=False)

    # Load model and scaler immediately — these are always safe to restore
    model.load_state_dict(state["model_state"])
    scaler.load_state_dict(state["scaler_state"])

    # Rebuild config from saved state
    c = state["config"]
    saved_config = TrainConfig(
        manifest_path=Path(c["manifest_path"]),
        output_dir=Path(c["output_dir"]),
        image_size=c.get("image_size", 380),
        batch_size=c.get("batch_size", 32),
        num_workers=c.get("num_workers", 0),
        warmup_epochs=c.get("warmup_epochs", 3),
        finetune_epochs=c.get("finetune_epochs", 12),
        lr_head=c.get("lr_head", 5e-4),
        lr_finetune=c.get("lr_finetune", 1.5e-4),
        weight_decay=c.get("weight_decay", 1e-5),
        patience=c.get("patience", 4),
        label_smoothing=c.get("label_smoothing", 0.05),
        amp=c.get("amp", True),
        subset_fraction=c.get("subset_fraction", 1.0),
        seed=c.get("seed", 42),
    )

    print(f"  Resuming from epoch {state['epoch']}, phase={state['phase']}, "
          f"phase_epoch={state['phase_epoch']}")
    print(f"  Previous best accuracy: {state['best_acc']:.4f}")
    print(f"  Previous epochs logged: {len(state.get('log_records', []))}")

    return (
        state["epoch"],
        state["phase"],
        state["phase_epoch"],
        state["best_acc"],
        state.get("log_records", []),
        saved_config,
        state["num_classes"],
        state["optimizer_state"],   # ← returned for deferred loading
        state["scheduler_state"],   # ← returned for deferred loading
    )

# ─── Main Training Loop ──────────────────────────────────────────────────────

def train(
    config: TrainConfig,
    num_classes: int | None = None,
    resume_from: str | Path | None = None,
) -> Path:
    """Train model in two phases. If resume_from is given, load state and continue."""
    _set_seed(config.seed)
    if num_classes is None:
        num_classes = num_classes_from_manifest(config.manifest_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = config.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    resume_path = Path(resume_from) if resume_from else config.output_dir / RESUME_CKPT

    # Load datasets
    _print_box("Loading Datasets")
    train_ds = PlantDiseaseDataset.from_manifest(
        manifest_path=config.manifest_path,
        split="train",
        transform=build_transforms(image_size=config.image_size, train=True),
    )
    val_ds = PlantDiseaseDataset.from_manifest(
        manifest_path=config.manifest_path,
        split="val",
        transform=build_transforms(image_size=config.image_size, train=False),
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train/val splits are empty.")

    train_ds = _subset_if_needed(train_ds, config.subset_fraction)
    val_ds = _subset_if_needed(val_ds, config.subset_fraction)

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Classes: {num_classes}")

    # Data loaders
    sampler = _build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(f"\n  {_color('⚠ WARNING: CUDA not available — training on CPU (very slow!)', _RED)}")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  {_color(f'✓ Using GPU: {gpu_name} ({gpu_mem:.1f} GB)', _GREEN)}")

    # Model
    _print_box("Building Model")
    model = build_model(num_classes=num_classes, pretrained=True).to(device)
    trainable = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: EfficientNet-B3")
    print(f"  Parameters:   {total_params:,} total ({trainable:,} trainable initially)")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")
    stopper = EarlyStopping(patience=config.patience)

    log_records: list[dict[str, Any]] = []
    best_acc = -1.0
    best_ckpt = config.output_dir / "best.pt"
    class_names = load_label_map()

    # Create placeholders for optimizer/scheduler so resume loader can reference them
    # (they will be properly created in each phase block with correct parameters)
   
    # ── Attempt resume ────────────────────────────────────────────────────
    # Pass only model and scaler — optimizer/scheduler state is returned
    # for deferred loading AFTER the real optimizer is built per-phase
    resume_data = _load_training_state(
        resume_path, model, scaler, device
    ) if resume_path.exists() else None

    if resume_data is not None:
        saved_epoch, saved_phase, saved_phase_epoch, best_acc, \
            log_records, saved_config, saved_nc, \
            saved_opt_state, saved_sched_state = resume_data
        num_classes = saved_nc
        print(f"\n  {_color('✓ Training state loaded — resuming...', _GREEN)}")
        print(f"  Resuming from epoch {saved_epoch}, phase={saved_phase}, "
              f"phase_epoch={saved_phase_epoch}")
        print(f"  Best accuracy so far: {best_acc:.4f}")
        print(f"  Log records: {len(log_records)} epochs\n")
    else:
        saved_phase = ""
        saved_phase_epoch = 0
        saved_opt_state = None
        saved_sched_state = None
    checkpoint_phase = saved_phase  # preserve original before line 599 overwrites it    

    # ── Phase 1: Warmup ──────────────────────────────────────────────────
   # ── Phase 1: Warmup ──────────────────────────────────────────────────
    if saved_phase in ("", "warmup"):
        _print_box("PHASE 1: WARMUP — Training Classifier Head (Backbone Frozen)")
        model.freeze_backbone()
        optimizer = AdamW(model.head.parameters(), lr=config.lr_head, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(config.warmup_epochs, 1))

        # Now that the real optimizer is built, restore its state if resuming warmup
        if saved_phase == "warmup" and saved_opt_state is not None:
            optimizer.load_state_dict(saved_opt_state)
            scheduler.load_state_dict(saved_sched_state)

        start_epoch = saved_phase_epoch if saved_phase == "warmup" else 0

        for epoch in range(start_epoch, config.warmup_epochs):
            epoch_start = time.time()
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = _run_epoch(
                model, train_loader, criterion, optimizer, scaler, device, config.amp,
                desc=f"Train (warmup {epoch+1}/{config.warmup_epochs})",
            )
            val_result = _evaluate_loader(
                model, val_loader, criterion, scaler, device, config.amp,
                class_names=class_names, num_classes=num_classes,
            )
            scheduler.step()

            is_best = val_result["val_acc"] > best_acc
            if is_best:
                best_acc = val_result["val_acc"]
                torch.save({
                    "model_state": model.state_dict(), "val_acc": val_result["val_acc"],
                    "epoch": epoch + 1, "phase": "warmup",
                }, best_ckpt)

            elapsed = time.time() - epoch_start
            stats = {
                "phase": "warmup", "epoch": epoch + 1,
                "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
                **val_result,
            }
            log_records.append(stats)
            _print_epoch_summary("warmup", epoch + 1, config.warmup_epochs, current_lr,
                                stats, elapsed, model, is_best, device)

            # Save resume checkpoint after each warmup epoch
            _save_training_state(
                resume_path, model, optimizer, scheduler, scaler,
                epoch + 1, "warmup", epoch + 1, best_acc, log_records, config, num_classes,
            )

        saved_phase = "finetune"  # Move to finetune after warmup completes

    # ── Phase 2: Full fine-tuning ────────────────────────────────────────
    if not resume_data or checkpoint_phase == "warmup":
        if not resume_data or saved_phase != "finetune":
            _print_box("PHASE 2: FINETUNE — Full Network Training")
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=config.lr_finetune, weight_decay=config.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(config.finetune_epochs, 1))
            start_epoch = 0
        else:
            # Resuming mid-finetune — unfreeze backbone and restore optimizer state
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=config.lr_finetune, weight_decay=config.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(config.finetune_epochs, 1))
            optimizer.load_state_dict(saved_opt_state)
            scheduler.load_state_dict(saved_sched_state)
            start_epoch = saved_phase_epoch

        for epoch in range(start_epoch, config.finetune_epochs):
            epoch_start = time.time()
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = _run_epoch(
                model, train_loader, criterion, optimizer, scaler, device, config.amp,
                desc=f"Train (finetune {epoch+1}/{config.finetune_epochs})",
            )
            val_result = _evaluate_loader(
                model, val_loader, criterion, scaler, device, config.amp,
                class_names=class_names, num_classes=num_classes,
            )
            scheduler.step()

            is_best = val_result["val_acc"] > best_acc
            if is_best:
                best_acc = val_result["val_acc"]
                torch.save({
                    "model_state": model.state_dict(), "val_acc": val_result["val_acc"],
                    "epoch": epoch + 1, "phase": "finetune",
                }, best_ckpt)

            elapsed = time.time() - epoch_start
            stats = {
                "phase": "finetune", "epoch": epoch + 1,
                "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
                **val_result,
            }
            log_records.append(stats)
            _print_epoch_summary("finetune", epoch + 1, config.finetune_epochs, current_lr,
                                stats, elapsed, model, is_best, device)

            # Save resume checkpoint after each finetune epoch
            _save_training_state(
                resume_path, model, optimizer, scheduler, scaler,
                epoch + 1, "finetune", epoch + 1, best_acc, log_records, config, num_classes,
            )

            if stopper.step(val_result["val_acc"]):
                print(f"  {_color('⏹ Early stopping triggered — no improvement for {config.patience} epochs', _YELLOW)}")
                break

    # ── Finalize ─────────────────────────────────────────────────────────
    (logs_dir / "train_log.json").write_text(json.dumps(log_records, indent=2), encoding="utf-8")

    # Remove resume checkpoint since training is done
    if resume_path.exists():
        resume_path.unlink()
        print(f"  {_color('🧹 Resume checkpoint cleaned up', _GRAY)}")

    best_log = max(log_records, key=lambda x: x.get("val_acc", 0))
    _print_box("TRAINING COMPLETE")
    print(f"  Best checkpoint: {best_ckpt}")
    print(f"  Best epoch:      {best_log['epoch']} ({best_log['phase']})")
    print(f"  Val accuracy:    {best_log['val_acc']:.4f}")
    print(f"  Macro F1:        {best_log.get('val_macro_f1', 0):.4f}")
    print(f"  Weighted F1:     {best_log.get('val_weighted_f1', 0):.4f}")
    print(f"  Train loss:      {best_log.get('train_loss', 0):.4f}")
    print(f"  Val loss:        {best_log.get('val_loss', 0):.4f}")

    _print_box("Exporting for Deployment")
    _export_for_deployment(model, best_ckpt, config.output_dir, device)

    return best_ckpt


# ─── CLI Entrypoint ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Train Plant Disease Classifier")
    parser.add_argument("--manifest-path", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "manifest.json")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--lr-finetune", type=float, default=1.5e-4)
    parser.add_argument("--subset-fraction", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from saved state in output directory")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 1 epoch each, 5% of data, batch=8")
    return parser.parse_args()


def _handle_interrupt(sig=None, frame=None):
    """Handle Ctrl+C gracefully during training."""
    print(f"\n  {_color('⏸ Training paused! State saved. Run with --resume to continue.', _YELLOW)}")
    print(f"  {_color('   Use Ctrl+C again to force quit.', _GRAY)}")
    raise SystemExit(0)


def main() -> None:
    """Entrypoint for training."""
    import signal
    signal.signal(signal.SIGINT, _handle_interrupt)

    args = parse_args()
    if args.smoke:
        warmup_epochs, finetune_epochs = 1, 1
        subset_fraction, batch_size = 0.05, min(args.batch_size, 8)
        print(f"\n  {_color('SMOKE TEST MODE — 1 epoch each, 5% data', _YELLOW)}\n")
    else:
        warmup_epochs = args.warmup_epochs
        finetune_epochs = args.finetune_epochs
        subset_fraction = args.subset_fraction
        batch_size = args.batch_size

    config = TrainConfig(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        batch_size=batch_size,
        warmup_epochs=warmup_epochs,
        finetune_epochs=finetune_epochs,
        subset_fraction=subset_fraction,
    )
    nc = num_classes_from_manifest(args.manifest_path)

    resume_path = config.output_dir / RESUME_CKPT if args.resume else None

    print(f"\n{'=' * 72}")
    print(f"  PLANT DISEASE DETECTOR — Training Pipeline")
    print(f"{'=' * 72}")
    if resume_path and resume_path.exists():
        print(f"  {_color('📂 RESUME MODE — Loading previous training state', _CYAN)}")
    print(f"  Config: {nc} classes, {config.image_size}x{config.image_size} images")
    print(f"  Phase 1: warmup ({config.warmup_epochs} epochs, backbone frozen)")
    print(f"  Phase 2: finetune ({config.finetune_epochs} epochs, full model)")
    print(f"  Batch: {config.batch_size} | LR head: {config.lr_head} | LR ft: {config.lr_finetune}")
    print(f"  Resume: {'Yes' if args.resume else 'No'} "
          f"{'(state found)' if resume_path and resume_path.exists() else '(no state)' if args.resume else ''}")
    print(f"  {'=' * 72}")
    print(f"  Press Ctrl+C to pause training — state will be saved automatically")
    print(f"{'=' * 72}\n")

    ckpt = train(config, num_classes=nc, resume_from=resume_path)
    print(f"\n  {_color('✓ Training complete!', _GREEN)}")
    print(f"  Best checkpoint: {ckpt}")
    print(f"  Saved to: {config.output_dir}\n")


if __name__ == "__main__":
    main()