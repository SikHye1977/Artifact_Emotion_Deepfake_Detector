"""Wrapper for an unmodified, pinned official AASIST checkout."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import torch
from torch import Tensor, nn

from .interfaces import ModelOutput


class AASISTWrapper(nn.Module):
    """Expose official embedding/logits and a configured fake probability."""

    def __init__(self, source_root: Path, model_args: dict, *, fake_class_index: int):
        super().__init__()
        if fake_class_index not in (0, 1):
            raise ValueError("Official AASIST has two class logits; fake_class_index must be 0 or 1")
        self.fake_class_index = fake_class_index
        self.backbone = _load_official_model(source_root, model_args)

    def forward(self, waveform: Tensor) -> ModelOutput:
        embedding, logits = self.backbone(waveform)
        fake_probability = torch.softmax(logits, dim=-1)[..., self.fake_class_index]
        return ModelOutput(features=embedding, logits=logits, fake_probability=fake_probability)


def _load_official_model(source_root: Path, model_args: dict) -> nn.Module:
    """Import `models.AASIST.Model` from the caller-verified vendor checkout."""
    source_root = Path(source_root).resolve()
    if not (source_root / "models" / "AASIST.py").is_file():
        raise FileNotFoundError(f"Official AASIST source missing: {source_root}")
    source_text = str(source_root)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        return import_module("models.AASIST").Model(model_args)
    finally:
        if inserted:
            sys.path.remove(source_text)


def build_aasist(source_root: Path, model_args: dict, *, fake_class_index: int) -> AASISTWrapper:
    """Build without loading an official checkpoint or any waveform data."""
    return AASISTWrapper(source_root, model_args, fake_class_index=fake_class_index)
