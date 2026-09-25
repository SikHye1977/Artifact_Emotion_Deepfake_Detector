"""Strict modality target selection and losses."""
import torch
from torch.nn import functional as F


def select_targets(batch, index, device):
    mask = batch['label_mask'][:, index].bool()
    labels = batch['labels'][:, index]
    if not mask.any():
        raise ValueError('No labeled samples in batch')
    y = labels[mask].to(device)
    if not ((y == 0) | (y == 1)).all():
        raise ValueError('Invalid binary targets')
    return mask, y


def modality_loss(logits, targets, task):
    expected = (len(targets),) if task == 'video' else (len(targets),2)
    if tuple(logits.shape) != expected:
        raise ValueError(f'Logits shape {logits.shape} != {expected}')
    return F.binary_cross_entropy_with_logits(logits, targets.float()) if task == 'video' else F.cross_entropy(logits,targets.long())


def binary_logit_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits,targets)
