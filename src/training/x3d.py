"""Reusable X3D training entry and strict model input adapter."""
from .losses import select_targets


def step(model,batch,device):
    mask,y=select_targets(batch,1,device)
    x=batch['video'][mask].to(device)
    if tuple(x.shape[1:])!=(3,128,256,256):raise ValueError('X3D requires [B,3,128,256,256]')
    return model(x),y,[s for s,m in zip(batch['sample_id'],mask.tolist()) if m]


def run_x3d_training(config_path,project_root,*,dry_run=False,resume_checkpoint=None,max_epochs=None):
    from .trainer import run_training
    return run_training(config_path,project_root,'video',step,dry_run=dry_run,resume_checkpoint=resume_checkpoint,max_epochs=max_epochs)
