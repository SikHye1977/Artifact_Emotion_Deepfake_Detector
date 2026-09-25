"""Reusable distributed-training contracts shared by diagnostics and formal runs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch

from src.datasets.distributed import GlobalBalancedSampler, UnpaddedDistributedSampler


def accumulation_scale(loss, micro_count):
    """Scale a micro-batch loss by the actual accumulation count."""
    if micro_count < 1:
        raise ValueError('micro_count must be positive')
    return loss / float(micro_count)


def expected_accumulation_groups(num_samples, micro_batch, accumulation_steps, world_size):
    if min(num_samples, micro_batch, accumulation_steps, world_size) < 1:
        raise ValueError('invalid accumulation parameters')
    global_micro = micro_batch * world_size
    groups, remainder = divmod(num_samples, global_micro * accumulation_steps)
    result = [accumulation_steps] * groups
    if remainder:
        result.append((remainder + global_micro - 1) // global_micro)
    return result


def collect_exact_predictions(parts, expected_ids):
    rows = [row for part in parts for row in part]
    ids = [row['sample_id'] for row in rows]
    expected = list(expected_ids)
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate dev sample_id')
    if set(ids) != set(expected) or len(ids) != len(expected):
        raise ValueError('dev sample coverage mismatch')
    return rows


def choose_best_checkpoint(current, candidate, *, metric='roc_auc', tie_metric='eer'):
    if current is None:
        return True
    if candidate[metric] > current[metric]:
        return True
    return candidate[metric] == current[metric] and candidate[tie_metric] < current[tie_metric]


def atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def sampler_contract(labels, *, rank, world_size, seed=42, batch_size=2):
    sampler = GlobalBalancedSampler(labels, rank, world_size, seed=seed, batch_size=batch_size)
    return sampler, sampler.summary()


__all__ = [
    'GlobalBalancedSampler', 'UnpaddedDistributedSampler', 'accumulation_scale',
    'expected_accumulation_groups', 'collect_exact_predictions',
    'choose_best_checkpoint', 'atomic_torch_save', 'sampler_contract'
]
