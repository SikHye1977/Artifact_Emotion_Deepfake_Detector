"""Stable single-model output contract for all downstream branches."""
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class ModelOutput:
    """Feature vector and unnormalized class logits from one model."""

    features: Tensor
    logits: Tensor
    fake_probability: Tensor | None = None


class SingleModel(Protocol):
    """A wrapper accepts already decoded media and never reads paths itself."""

    def __call__(self, inputs: Tensor, **kwargs: object) -> ModelOutput: ...