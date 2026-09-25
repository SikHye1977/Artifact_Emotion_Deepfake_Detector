"""Typed branch outputs consumed by fusion modules."""
from dataclasses import dataclass
from torch import Tensor
from src.models.interfaces import ModelOutput


@dataclass(frozen=True)
class ArtifactOutput:
    video: ModelOutput
    audio: ModelOutput


@dataclass(frozen=True)
class BranchOutput:
    score: Tensor
    features: Tensor | None = None
