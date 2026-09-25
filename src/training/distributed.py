"""Bounded rank error exchange and durable DDP recovery state."""
from datetime import timedelta
from pathlib import Path
import hashlib
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from src.utils.config import sampler_policy_sha256


def gather(value,group):
    result=[None]*dist.get_world_size(group)
    dist.all_gather_object(result,value,group=group)
    return result


def agree(error,group):
    errors=gather(error,group)
    if any(errors):raise RuntimeError('Rank errors: '+repr(errors))


def guarded(fn,group):
    result=None;error=None
    try:result=fn()
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    agree(error,group)
    return result


def visible_physical_devices(value=None):
    value=os.environ.get('CUDA_VISIBLE_DEVICES','') if value is None else value
    return [int(x) for x in value.split(',') if x.strip()]


def initialize():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    world=int(os.environ['WORLD_SIZE'])
    if world not in (2,3,4):raise ValueError('Distributed training requires two, three or four ranks')
    visible=visible_physical_devices()
    if world==3 and visible != [1,2,3]: raise ValueError('Three-GPU diagnostic requires CUDA_VISIBLE_DEVICES=1,2,3')
    if world==2 and visible != [1,2]: raise ValueError('Two-GPU formal training requires CUDA_VISIBLE_DEVICES=1,2')
    if world==4 and visible and visible != [0,1,2,3]: raise ValueError('Four-GPU diagnostic requires physical devices 0,1,2,3')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=120))
    return rank,torch.device('cuda',local),control


def state_digest(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def local_rng(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))


def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'],device)


def save_recovery(path,model,optimizer,scheduler,scaler,sampler,policy,provenance,*,global_step,attempts,device,group):
    # BN running statistics are rank-local during training; rank 0 is canonical for recovery/evaluation.
    for buffer in model.buffers():dist.broadcast(buffer,src=0)
    states=gather(dict(rng=local_rng(device),sampler=sampler.state_dict(),policy=policy.state_dict()),group)
    error=None
    if dist.get_rank()==0:
        try:
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),epoch=sampler.epoch,batch_offset=sampler.batch_offset,global_step=global_step,
                optimizer_attempts=attempts,rank_states=states,provenance=provenance,world_size=dist.get_world_size(),
                global_effective_batch_size=provenance['config']['training']['global_effective_batch_size'],
                sampler_policy=provenance['sampler_policy'],sampler_policy_sha256=provenance['sampler_policy_sha256'],
                best_dev_metric=None)
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        except Exception as exc:error=repr(exc)
    agree(error,group)


def load_recovery(path,model,optimizer,scheduler,scaler,sampler,policy,provenance,device,group):
    def load():
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['provenance']!=provenance or state['world_size']!=dist.get_world_size() or state.get('sampler_policy')!=provenance.get('sampler_policy') or state.get('sampler_policy_sha256')!=sampler_policy_sha256(provenance['sampler_policy']):raise ValueError('Resume provenance/world size/sampler policy mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        local=state['rank_states'][dist.get_rank()]
        sampler.load_state_dict(local['sampler']);policy.load_state_dict(local['policy']);restore_rng(local['rng'],device)
        return state['global_step'],state['optimizer_attempts']
    steps=guarded(load,group)
    digests=gather(state_digest(model),group)
    agree(None if len(set(digests))==1 else 'Restored model differs across ranks',group)
    return steps,digests[0]
