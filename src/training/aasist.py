"""Reusable fresh AASIST training entry with project label semantics."""
from .losses import select_targets


def step(model,batch,device):
    if model.fake_class_index!=1:raise ValueError('Expected fake_class_index=1')
    mask,y=select_targets(batch,2,device)
    x=batch['audio'][mask].to(device)
    if tuple(x.shape[1:])!=(284672,):raise ValueError('AASIST requires [B,284672]')
    if (batch['audio_lengths']>284672).any():raise ValueError('Truncation prohibited')
    return model(x),y,[s for s,m in zip(batch['sample_id'],mask.tolist()) if m]


def run_aasist_training(config_path,project_root,*,dry_run=False,resume_checkpoint=None):
    from .trainer import run_training
    return run_training(config_path,project_root,'audio',step,dry_run=dry_run,resume_checkpoint=resume_checkpoint)
