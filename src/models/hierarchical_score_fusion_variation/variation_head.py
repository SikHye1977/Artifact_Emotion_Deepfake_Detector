"""Classify statistics of local emotion changes, not mean emotion embeddings."""
from src.models.hierarchical_score_fusion.emotion_deepfake_head import EmotionDeepfakeHead


class VariationHead(EmotionDeepfakeHead):
    def __init__(self, modality, hidden_dim=64, dropout=.2, input_dim=27):
        if modality not in ('video','audio'): raise ValueError('Unknown modality')
        super().__init__(input_dim,hidden_dim,dropout)
