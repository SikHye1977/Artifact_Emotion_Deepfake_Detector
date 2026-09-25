"""Frozen emotion extractors + two trainable mean-feature fake classifiers."""
import numpy as np
import torch
from torch import nn
from src.models.hierarchical_score_fusion.emotion_deepfake_head import EmotionDeepfakeHead
from src.fusion.hierarchical_score_fusion.hierarchical_score_fusion import probabilistic_or


class HierarchicalEmotionBranch(nn.Module):
    def __init__(self, hsemotion=None, acrnn=None, video_dim=1280, audio_dim=256,
                 hidden_dim=64, dropout=0.2):
        super().__init__()
        # HSEmotion's supplied wrapper is not an nn.Module.
        self.hsemotion = hsemotion
        self.acrnn = acrnn
        if acrnn is not None:
            acrnn.requires_grad_(False).eval()
        if hsemotion is not None:
            hsemotion.recognizer.model.requires_grad_(False).eval()
        self.video_head = EmotionDeepfakeHead(video_dim, hidden_dim, dropout)
        self.audio_head = EmotionDeepfakeHead(audio_dim, hidden_dim, dropout)

    def train(self, mode=True):
        super().train(mode)
        if self.acrnn is not None:
            self.acrnn.eval()
        if self.hsemotion is not None:
            self.hsemotion.recognizer.model.eval()
        return self

    def extract_video(self, face_rgb):
        if self.hsemotion is None:
            raise RuntimeError('Attach the supplied HSEmotion extractor')
        f = np.asarray(self.hsemotion.extract(face_rgb)['features'], dtype=np.float32).reshape(-1)
        if f.shape != (self.video_head.input_dim,) or not np.isfinite(f).all():
            raise ValueError('Unexpected HSEmotion embedding')
        return f

    def extract_audio(self, acoustic):
        if self.acrnn is None:
            raise RuntimeError('Attach the supplied ACRNN extractor')
        # The supplied ACRNN returns inference tensors. Convert before head training.
        with torch.no_grad():
            f = self.acrnn(acoustic)['features'].detach().cpu().numpy().copy()
        if f.shape != (len(acoustic), self.audio_head.input_dim) or not np.isfinite(f).all():
            raise ValueError('Unexpected ACRNN embedding')
        return f

    def forward(self, video_features, audio_features, video_mask=None, audio_mask=None):
        lv = self.video_head(video_features, video_mask)
        la = self.audio_head(audio_features, audio_mask)
        ev, ea = lv.sigmoid(), la.sigmoid()
        return {'video_logits': lv, 'audio_logits': la, 'video_score': ev,
                'audio_score': ea, 'score_emotion': probabilistic_or(ev, ea)}
