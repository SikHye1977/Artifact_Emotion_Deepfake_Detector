"""Independent modality heads: raw embeddings -> 128-d projections -> vector gate."""
import torch
from torch import nn


class GatedBranch(nn.Module):
    def __init__(self, artifact_dim, emotion_dim, dropout=0.2):
        super().__init__()
        self.artifact_projection = nn.Sequential(nn.Linear(artifact_dim, 128), nn.LayerNorm(128))
        self.emotion_projection = nn.Sequential(nn.Linear(emotion_dim, 128), nn.LayerNorm(128))
        self.gate = nn.Linear(257, 128)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, artifact, emotion, emotion_valid):
        if artifact.ndim != 2 or emotion.ndim != 2 or len(artifact) != len(emotion):
            raise ValueError('Expected two embedding matrices with equal batch size')
        valid = emotion_valid.reshape(-1, 1).bool()
        if len(valid) != len(artifact):
            raise ValueError('Invalid emotion mask')
        a = self.artifact_projection(artifact)
        # Mask before projection: invalid NaNs must never reach parameter gradients.
        clean = torch.where(valid, emotion, torch.zeros_like(emotion))
        e = self.emotion_projection(clean)
        gate = torch.sigmoid(self.gate(torch.cat([a, e, valid.to(a.dtype)], dim=1)))
        gate = torch.where(valid, gate, torch.ones_like(gate))
        fused = gate * a + (1 - gate) * e
        return self.classifier(fused).squeeze(-1), gate


def score_or(video_probability, audio_probability):
    return 1 - (1 - video_probability) * (1 - audio_probability)
