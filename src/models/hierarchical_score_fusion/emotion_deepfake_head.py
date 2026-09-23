"""Trainable fake/real head; the emotion encoders remain frozen."""
import torch
from torch import nn


def masked_mean(features, mask):
    if features.ndim != 3 or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Expected features [B,T,D] and bool mask [B,T]')
    if not mask.any(dim=1).all():
        raise ValueError('No valid emotion observation; do not interpret missing data as real')
    if not torch.isfinite(features[mask]).all():
        raise ValueError('Nonfinite valid feature')
    # Multiplication by zero would retain NaN in invalid positions.
    return features.masked_fill(~mask.unsqueeze(-1), 0).sum(1) / mask.sum(1, keepdim=True)


class EmotionDeepfakeHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('scale', torch.ones(input_dim))
        self.register_buffer('standardizer_fitted', torch.tensor(False))
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    @torch.no_grad()
    def fit_standardizer(self, train_features):
        self._validate(train_features)
        self.mean.copy_(train_features.mean(0))
        scale = train_features.std(0, unbiased=False)
        self.scale.copy_(torch.where(scale < 1e-6, torch.ones_like(scale), scale))
        self.standardizer_fitted.fill_(True)

    def _validate(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim or len(x) == 0 or not torch.isfinite(x).all():
            raise ValueError(f'Expected nonempty finite [B,{self.input_dim}]')

    def forward(self, features, mask=None):
        x = masked_mean(features, mask) if mask is not None else features
        self._validate(x)
        if not bool(self.standardizer_fitted):
            raise RuntimeError('Fit standardizer on training data, or load a trained checkpoint')
        return self.classifier((x - self.mean) / self.scale).squeeze(-1)
