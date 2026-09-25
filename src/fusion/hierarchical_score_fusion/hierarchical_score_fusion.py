"""Parameter-free probabilistic-OR score fusion. Inputs are NOT logits."""
import torch
from torch import nn


def probabilistic_or(left, right):
    if left.ndim != 1 or left.shape != right.shape or left.device != right.device:
        raise ValueError('Aligned [B] scores on the same device required')
    for value in (left, right):
        if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError('Finite probabilities in [0,1] required')
    return 1 - (1 - left) * (1 - right)


class HierarchicalScoreFusion(nn.Module):
    def forward(self, score_artifact, score_emotion):
        return probabilistic_or(score_artifact, score_emotion)
