"""Checkpoint metadata validation boundary; no checkpoint is written on import."""
from pathlib import Path
from copy import deepcopy
from .config import sampler_policy_sha256


def checkpoint_metadata(config):
    """Metadata payload only: does not select, save, or load a checkpoint."""
    model = config['model']
    metadata = {'experiment_id': config['experiment']['id'], 'model': model['name']}
    if model['name'] == 'aasist':
        if model.get('class_mapping') != {'real': 0, 'fake': 1} or model.get('fake_class_index') != 1:
            raise ValueError('Fresh AASIST requires real=0, fake=1')
        metadata.update(class_mapping=deepcopy(model['class_mapping']), fake_class_index=1,
                        upstream_commit=config['third_party']['commit'])
    return metadata


def require_checkpoint(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


import os
import random
import numpy as np
import torch


def rng_state(generator=None):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                sampler=generator.get_state() if generator is not None else None)


def save_checkpoint(path, model, optimizer, scheduler, scaler, *, epoch, global_step,
                    best_dev_metric, provenance, generator=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),epoch=epoch,global_step=global_step,best_dev_metric=best_dev_metric,
        provenance=deepcopy(provenance),sampler_policy=deepcopy(provenance.get('sampler_policy')),
        sampler_policy_sha256=provenance.get('sampler_policy_sha256'),rng=rng_state(generator))
    temp=path.with_suffix('.tmp')
    torch.save(state,temp);os.replace(temp,path)


def load_checkpoint(path,model,optimizer,scheduler,scaler,*,provenance,generator=None):
    # Only load locally generated trusted training checkpoints (includes RNG objects).
    state=torch.load(require_checkpoint(path),map_location='cpu',weights_only=False)
    policy=provenance.get('sampler_policy')
    policy_ok=policy is None and state.get('sampler_policy') is None and state.get('sampler_policy_sha256') is None or (state.get('sampler_policy') == policy and state.get('sampler_policy_sha256') == sampler_policy_sha256(policy))
    if state['provenance'] != provenance or not policy_ok:
        raise ValueError('Resume config/source/weight/data hash mismatch')
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
    rng=state['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:torch.cuda.set_rng_state_all(rng['cuda'])
    if generator is not None and rng['sampler'] is not None:generator.set_state(rng['sampler'])
    return state
