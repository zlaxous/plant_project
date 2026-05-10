"""Generate Grad-CAM visualizations for model predictions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dataset import build_transforms
from model import build_model, load_checkpoint_weights
from paths import default_checkpoint_path, num_classes_from_manifest


def _overlay_heatmap(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * image_rgb + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    return overlay


def _gradcam_overlay_for_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 380,
) -> np.ndarray:
    """Compute Grad-CAM overlay array for one image (model in eval mode)."""
    target_layer = model.backbone.conv_head
    gradients: list[torch.Tensor] = []
    activations: list[torch.Tensor] = []

    def _save_gradients(
        _: torch.nn.Module, __: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        gradients.append(grad_output[0].detach())

    def _save_activations(
        _: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        activations.append(output.detach())

    handle_forward = target_layer.register_forward_hook(_save_activations)
    handle_backward = target_layer.register_full_backward_hook(_save_gradients)
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image.resize((image_size, image_size)))
        tensor = build_transforms(image_size=image_size, train=False)(image).unsqueeze(0).to(device)
        logits = model(tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, pred_idx]
        model.zero_grad(set_to_none=True)
        score.backward()

        grad = gradients[0][0]
        act = activations[0][0]
        weights = torch.mean(grad, dim=(1, 2))
        cam = torch.sum(weights[:, None, None] * act, dim=0)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np, (image_size, image_size))
        return _overlay_heatmap(image_np, cam_resized)
    finally:
        handle_forward.remove()
        handle_backward.remove()


def generate_gradcam(
    image_path: Path, checkpoint_path: Path, out_path: Path, manifest_path: Path | None = None
) -> None:
    """Create Grad-CAM overlay image for the top predicted class."""
    from paths import default_manifest_path

    mpath = manifest_path or default_manifest_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = num_classes_from_manifest(mpath)
    model = build_model(num_classes=nc, pretrained=False).to(device)
    load_checkpoint_weights(model, str(checkpoint_path))
    model.eval()

    overlay = _gradcam_overlay_for_image(image_path, model, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)


def generate_gradcam_grid(
    image_paths: list[Path],
    checkpoint_path: Path,
    out_path: Path,
    manifest_path: Path | None = None,
    ncols: int = 4,
    image_size: int = 380,
) -> None:
    """Save a grid of Grad-CAM overlays for multiple images."""
    from paths import default_manifest_path

    mpath = manifest_path or default_manifest_path()
    nc = num_classes_from_manifest(mpath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=nc, pretrained=False).to(device)
    load_checkpoint_weights(model, str(checkpoint_path))
    model.eval()

    tiles: list[Image.Image] = []
    for p in image_paths:
        arr = _gradcam_overlay_for_image(p, model, device, image_size=image_size)
        tiles.append(Image.fromarray(arr))

    n = len(tiles)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    grid_w = ncols * image_size
    grid_h = nrows * image_size
    grid = Image.new("RGB", (grid_w, grid_h), color=(32, 32, 32))
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, ncols)
        grid.paste(tile, (c * image_size, r * image_size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main() -> None:
    """Entrypoint for Grad-CAM generation."""
    parser = argparse.ArgumentParser(
        description="Grad-CAM for plant disease model. One image → overlay; multiple → grid."
    )
    parser.add_argument(
        "--image", type=Path, nargs="+", required=True, help="One or more input images"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--ncols", type=int, default=4)

    args = parser.parse_args()
    ckpt = args.checkpoint or default_checkpoint_path()
    if len(args.image) == 1:
        generate_gradcam(args.image[0], ckpt, args.output, manifest_path=args.manifest_path)
        print(f"Saved Grad-CAM to {args.output}")
    else:
        generate_gradcam_grid(
            list(args.image),
            ckpt,
            args.output,
            manifest_path=args.manifest_path,
            ncols=args.ncols,
        )
        print(f"Saved Grad-CAM grid to {args.output}")


if __name__ == "__main__":
    main()
