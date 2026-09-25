"""Reuse trained X3D/AASIST; no artifact parameter is optimized here."""
import torch
from torch import nn
from src.fusion.hierarchical_score_fusion.hierarchical_score_fusion import probabilistic_or


class HierarchicalArtifactBranch(nn.Module):
    def __init__(self, x3d, aasist):
        super().__init__()
        self.x3d = x3d.requires_grad_(False).eval()
        self.aasist = aasist.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, video, waveform):
        if video.ndim != 5 or video.shape[1:3] != (3, 128):
            raise ValueError('X3D requires [B,3,128,H,W]')
        if waveform.ndim != 2 or len(video) != len(waveform):
            raise ValueError('Aligned waveform [B,L] required')
        self.eval()
        av = self.x3d(video).fake_probability
        aa = self.aasist(waveform).fake_probability
        return {'video_score': av, 'audio_score': aa,
                'score_artifact': probabilistic_or(av, aa)}
