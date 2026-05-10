"""Model definition for plant disease classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for PlantDiseaseModel."""

    num_classes: int = 38
    dropout: float = 0.3
    pretrained: bool = True


class PlantDiseaseModel(nn.Module):
    """EfficientNet-B3 backbone with custom classification head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=config.pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = int(self.backbone.num_features)
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(512, config.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute class logits from RGB image batch."""
        features = self.backbone(images)
        return self.head(features)

    @torch.inference_mode()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self(images)
        return torch.softmax(logits, dim=1)

    @torch.inference_mode()
    def predict_topk(self, images: torch.Tensor, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k probabilities and class indices."""
        probs = self.predict_proba(images)
        return torch.topk(probs, k=k, dim=1)

    def freeze_backbone(self) -> None:
        """Freeze backbone for transfer-learning warmup phase."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def to_device(self, device: torch.device | str) -> PlantDiseaseModel:
        """Move model to target device and return self."""
        self.to(device)
        return self

    def extra_repr(self) -> str:
        return f"num_classes={self.config.num_classes}, dropout={self.config.dropout}"


def build_model(
    num_classes: int = 38, pretrained: bool = True, dropout: float = 0.3
) -> PlantDiseaseModel:
    """Factory function to build classifier model."""
    return PlantDiseaseModel(
        ModelConfig(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def model_summary_text(model: nn.Module) -> str:
    """Build a simple text summary independent of optional summary libs."""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = count_trainable_parameters(model)
    lines = [
        f"Model: {model.__class__.__name__}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Frozen parameters: {total_params - trainable_params:,}",
    ]
    if isinstance(model, PlantDiseaseModel):
        lines.append(f"Classes: {model.config.num_classes}")
        lines.append(f"Dropout: {model.config.dropout}")
    return "\n".join(lines) + "\n"


def load_checkpoint_weights(model: nn.Module, checkpoint_path: str) -> dict[str, Any]:
    """Load model weights from checkpoint and return metadata."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}
