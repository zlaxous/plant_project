"""Export trained classifier to ONNX and TorchScript with verification."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from model import build_model, load_checkpoint_weights
from paths import default_manifest_path, num_classes_from_manifest


def export_onnx(
    checkpoint_path: Path,
    onnx_path: Path,
    manifest_path: Path | None = None,
    image_size: int = 380,
    opset: int = 17,
) -> None:
    """Export model to ONNX (batch size 1, fixed HxW).

    Uses the dynamo/torch.export-based exporter (newer ONNX path).
    """
    mpath = manifest_path or default_manifest_path()
    nc = num_classes_from_manifest(mpath)
    model = build_model(num_classes=nc, pretrained=False)
    load_checkpoint_weights(model, str(checkpoint_path))
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch's dynamo ONNX path can emit a noisy pytree FutureWarning on some versions.
    # It's not actionable for this project, so we silence it locally.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=None,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=True,
        )


def export_torchscript(
    checkpoint_path: Path,
    ts_path: Path,
    manifest_path: Path | None = None,
    image_size: int = 380,
) -> None:
    """Export a stable serialized program for inference.

    TorchScript is deprecated in recent PyTorch versions; we instead save a
    `torch.export` program (still written to a `.pt` path for backward
    compatibility with existing tooling/tests).
    """
    mpath = manifest_path or default_manifest_path()
    nc = num_classes_from_manifest(mpath)
    model = build_model(num_classes=nc, pretrained=False)
    load_checkpoint_weights(model, str(checkpoint_path))
    model.eval()
    example = (torch.randn(1, 3, image_size, image_size),)
    exported = torch.export.export(model, example)
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, str(ts_path))


def verify_onnx(
    onnx_path: Path,
    checkpoint_path: Path,
    manifest_path: Path | None = None,
    image_size: int = 380,
    rtol: float = 1e-3,
    atol: float = 1e-2,
) -> dict[str, Any]:
    """Compare ONNX Runtime output with PyTorch on a random tensor."""
    mpath = manifest_path or default_manifest_path()
    nc = num_classes_from_manifest(mpath)
    pytorch_model = build_model(num_classes=nc, pretrained=False)
    load_checkpoint_weights(pytorch_model, str(checkpoint_path))
    pytorch_model.eval()
    x = torch.randn(1, 3, image_size, image_size)
    with torch.inference_mode():
        torch_out = pytorch_model(x).numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: x.numpy()})[0]

    max_abs = float(np.max(np.abs(onnx_out - torch_out)))
    ok = bool(np.allclose(onnx_out, torch_out, rtol=rtol, atol=atol))
    return {
        "onnx_path": str(onnx_path),
        "max_abs_diff": max_abs,
        "outputs_match": ok,
        "rtol": rtol,
        "atol": atol,
    }


def main() -> None:
    """CLI: export ONNX + TorchScript and verify ONNX."""
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX and TorchScript.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "exports")
    parser.add_argument("--image-size", type=int, default=380)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_p = out_dir / "model.onnx"
    ts_p = out_dir / "model_exported.pt2"

    export_onnx(
        args.checkpoint, onnx_p, manifest_path=args.manifest_path, image_size=args.image_size
    )
    print(f"Wrote ONNX: {onnx_p}")
    export_torchscript(
        args.checkpoint, ts_p, manifest_path=args.manifest_path, image_size=args.image_size
    )
    print(f"Wrote exported program: {ts_p}")

    report = verify_onnx(
        onnx_p, args.checkpoint, manifest_path=args.manifest_path, image_size=args.image_size
    )
    print(report)
    if not report["outputs_match"]:
        raise SystemExit(
            f"ONNX verification failed: max_abs_diff={report['max_abs_diff']}. "
            "Try a different opset or disable constant folding."
        )


if __name__ == "__main__":
    main()
