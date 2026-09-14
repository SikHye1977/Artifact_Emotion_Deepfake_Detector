"""Transform specifications shared by frozen media datasets.

Concrete training augmentation is intentionally deferred until the Phase 1
model recipe is fixed; this module never resolves a machine-local data path.
"""
from dataclasses import dataclass
import torch


def pad_audio(waveform, target_num_samples):
    if waveform.ndim != 1 or target_num_samples < 1:
        raise ValueError('Expected mono waveform and positive target')
    if waveform.numel() > target_num_samples:
        raise ValueError('Truncation is prohibited')
    output = waveform.new_zeros(target_num_samples)
    output[:waveform.numel()] = waveform
    return output, torch.arange(target_num_samples, device=waveform.device) < waveform.numel()


def balanced_weights(rows, label_key):
    if label_key not in ('video_label', 'audio_label'):
        raise ValueError('Specify modality label')
    counts = {y: sum(row[label_key] == y for row in rows) for y in (0, 1)}
    if not all(counts.values()):
        raise ValueError('Both classes required')
    return torch.tensor([0.0 if row[label_key] is None else 1/counts[row[label_key]]
                         for row in rows], dtype=torch.double)


@dataclass(frozen=True)
class VideoTransformSpec:
    frames: int
    crop_size: int
    resize_short: int
    mean: tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: tuple[float, float, float] = (0.225, 0.225, 0.225)


@dataclass(frozen=True)
class AudioTransformSpec:
    sample_rate: int = 16_000
    padding: str = "deferred"
