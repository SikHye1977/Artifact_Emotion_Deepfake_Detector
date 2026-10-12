"""Independent modality training with source-dev selection and epoch resume."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import require,metrics,seed_all,cpu_state,save_torch,atomic_json,read,sha,Progress,status
from .data import SequenceDataset,collate
from .model import EmotionFlowHead

@torch.inference_mode()
def predict(head,data,mod,device,batch_size,attention=False,progress=False):
    ds=SequenceDataset(data,mod);loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    scores=np.full(len(data['sample_id']),np.nan,np.float64)
    weights=np.zeros(len(data['valid']),np.float32) if attention else None
    head.eval();bar=Progress('PREDICT '+mod,len(ds)) if progress else None;done=0
    for e,p,m,_,indices,positions in loader:
        logits,alpha=head(e.to(device),p.to(device),m.to(device));scores[indices]=logits.sigmoid().cpu().numpy()
        if attention:
            a=alpha.cpu().numpy()
            for j,pos in enumerate(positions):weights[pos]=a[j,:len(pos)]
        done+=len(indices)
        if bar:bar.update(done)
    require(np.isfinite(scores[ds.indices]).all(),'Nonfinite predicted score')
    return scores,weights


def fit(train,dev,mod,folder,cfg,identity,interrupt_after_epoch=None):
    for key in ('sample_id','group'):
        require(not(set(train[key])&set(dev[key])),'Train/dev overlap: '+key)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    completed=folder/'COMPLETED.json'
    if completed.exists():
        old=read(completed);require(old['identity']==identity and sha(folder/'best.pt')==old['checkpoint_sha256'],'Completed head changed')
        status('TRAIN',mod+' completed head verified');return old
    device=cfg['head_device'];seed=cfg['seed']+(mod=='audio');seed_all(seed)
    ds=SequenceDataset(train,mod,labeled=True);ix=1 if mod=='video' else 2
    keep=(dev['valid_counts']>0)&(dev['labels'][:,ix]>=0)
    require(set(train['labels'][ds.indices,ix])=={0,1} and set(dev['labels'][keep,ix])=={0,1},'Need both known valid source classes')
    head=EmotionFlowHead(mod).to(device);optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg['pos_weight'],device=device))
    start=0;best=-1.;best_state=None;best_epoch=0;history=[];last=folder/'last.pt'
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=True);require(state['identity']==identity,'Resume training identity changed')
        head.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
        if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
        start,history,best,best_state,best_epoch=state['epoch'],state['history'],state['best_auc'],state['best_model'],state['best_epoch']
    for epoch in range(start,cfg['epochs']):
        t=time.perf_counter();head.train();total=0.;count=0
        loader=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=0,collate_fn=collate,
            generator=torch.Generator().manual_seed(seed+epoch))
        bar=Progress(f'TRAIN {mod} epoch {epoch+1}/{cfg["epochs"]}',len(loader),unit='batches')
        for step,(e,p,m,y,_,_) in enumerate(loader,1):
            optimizer.zero_grad(set_to_none=True)
            logits,_=head(e.to(device),p.to(device),m.to(device));loss=criterion(logits,y.to(device))
            require(bool(torch.isfinite(loss)),'Nonfinite training loss');loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(y);count+=len(y);bar.update(step,f'loss={total/count:.6f}')
        probs,_=predict(head,dev,mod,device,cfg['batch_size']);report=metrics(dev['labels'][keep,ix],probs[keep])
        auc=report['roc_auc']
        if auc>best:best,best_epoch,best_state=auc,epoch+1,cpu_state(head)
        history.append(dict(epoch=epoch+1,loss=total/count,dev=report,seconds=time.perf_counter()-t))
        # This one atomic file contains the matching current AND best model and all stochastic state.
        save_torch(last,dict(identity=identity,epoch=epoch+1,model=cpu_state(head),optimizer=optimizer.state_dict(),
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
            best_auc=best,best_epoch=best_epoch,best_model=best_state,history=history))
        atomic_json(folder/'history.json',history)
        status('TRAIN',f'{mod} epoch={epoch+1}/{cfg["epochs"]} dev_auc={auc:.6f} best_epoch={best_epoch} best_auc={best:.6f} '
               f'epoch_seconds={history[-1]["seconds"]:.1f} remaining_epoch_seconds~{history[-1]["seconds"]*(cfg["epochs"]-epoch-1):.1f}')
        if interrupt_after_epoch==epoch+1:raise InterruptedError('Test interruption after atomic epoch commit')
    require(best_state is not None,'No model trained')
    save_torch(folder/'best.pt',dict(format='emotion_gru_temporal_attention_v1',identity=identity,modality=mod,
        model=best_state,best_epoch=best_epoch,dev_auc=best,config=cfg))
    result=dict(identity=identity,checkpoint_sha256=sha(folder/'best.pt'),best_epoch=best_epoch,dev_auc=best,
        epochs_completed=len(history),train_n=len(ds),dev_n=int(keep.sum()),seed=seed,pos_weight=cfg['pos_weight'])
    atomic_json(completed,result);return result


def load_head(folder,mod,identity,device):
    folder=Path(folder);done=read(folder/'COMPLETED.json')
    require(done['identity']==identity and sha(folder/'best.pt')==done['checkpoint_sha256'],'Head completion mismatch')
    state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    require(state['format']=='emotion_gru_temporal_attention_v1' and state['identity']==identity and state['modality']==mod,'Wrong temporal head')
    head=EmotionFlowHead(mod).to(device);head.load_state_dict(state['model']);return head.eval().requires_grad_(False)
