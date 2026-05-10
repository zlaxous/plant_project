"""CLI and helpers for image-level disease prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from dataset import build_transforms
from model import build_model, load_checkpoint_weights
from paths import (
    default_checkpoint_path,
    default_manifest_path,
    load_label_map,
    num_classes_from_manifest,
)


def load_model_for_inference(
    checkpoint_path: Path,
    manifest_path: Path | None = None,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, torch.device, dict[int, str], int]:
    """Load weights and return model (eval), device, label map, num_classes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpath = manifest_path or default_manifest_path()
    nc = num_classes_from_manifest(mpath)
    model = build_model(num_classes=nc, pretrained=False).to(device)
    load_checkpoint_weights(model, str(checkpoint_path))
    model.eval()
    labels = load_label_map()
    return model, device, labels, nc


def predict_image(
    image_path: Path,
    checkpoint_path: Path,
    manifest_path: Path | None = None,
    topk: int = 5,
    image_size: int = 380,
) -> dict[str, Any]:
    """Return top-k predictions for an image file."""
    model, device, labels, _nc = load_model_for_inference(checkpoint_path, manifest_path)
    image = Image.open(image_path).convert("RGB")
    tensor = build_transforms(image_size=image_size, train=False)(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs, indices = model.predict_topk(tensor, k=min(topk, model.config.num_classes))
    probs_l = probs[0].cpu().tolist()
    idx_l = indices[0].cpu().tolist()
    predictions = [
        {
            "class_id": int(i),
            "class_name": labels.get(int(i), str(i)),
            "probability": float(p),
        }
        for p, i in zip(probs_l, idx_l, strict=False)
    ]
    return {
        "image": str(image_path),
        "top_k": predictions,
        "top1": predictions[0] if predictions else None,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Predict plant disease class for an image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    ckpt = args.checkpoint or default_checkpoint_path()
    result = predict_image(args.image, ckpt, manifest_path=args.manifest_path, topk=args.topk)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
