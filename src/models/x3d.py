from __future__ import annotations

import torch
from torch import nn
from .interfaces import ModelOutput


class X3DDeepfakeClassifier(nn.Module):
    """Kinetics-pretrained X3D with a binary fake/real classification head."""

    def __init__(
        self,
        variant: str = "x3d_m",
        *,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if variant not in {"x3d_xs", "x3d_s", "x3d_m"}:
            raise ValueError(f"Unsupported X3D variant: {variant}")

        try:
            from pytorchvideo.models import hub
        except ImportError as error:
            raise RuntimeError(
                "Install the locked PyTorchVideo dependency before building X3D."
            ) from error

        self.backbone = getattr(hub, variant)(pretrained=pretrained)

        # Kinetics-400 classifier → one fake logit
        projection = self.backbone.blocks[-1].proj
        self.backbone.blocks[-1].proj = nn.Linear(
            projection.in_features,
            1,
        )

        # BCEWithLogitsLoss를 사용하므로 Softmax를 제거한다.
        self.backbone.blocks[-1].activation = nn.Identity()

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

            for parameter in self.backbone.blocks[-1].parameters():
                parameter.requires_grad = True

    def forward(self, video: torch.Tensor) -> ModelOutput:
        """
        Args:
            video: normalized RGB tensor with shape [B, 3, T, H, W].

        Returns:
            ModelOutput with pooled features [B,D], logits [B], and fake probability [B].
        """
        # Capture the classifier input without changing pretrained state-dict keys.
        captured = []
        handle = self.backbone.blocks[-1].proj.register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        )
        try:
            logits = self.backbone(video).flatten(start_dim=1).squeeze(1)
        finally:
            handle.remove()
        features = captured[0].reshape(video.shape[0], -1, captured[0].shape[-1]).mean(1)
        return ModelOutput(features=features, logits=logits, fake_probability=torch.sigmoid(logits))