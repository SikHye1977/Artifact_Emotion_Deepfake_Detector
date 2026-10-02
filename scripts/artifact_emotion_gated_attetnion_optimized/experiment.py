"""FAV feature extraction, independent modality training, locked probabilistic OR evaluation."""
from pathlib import Path
import argparse
import csv
import json
import random
import time
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.artifact_emotion_gated_attetnion_optimized.common import (
    implementation, read, write, sha, load_cache, disjoint, metrics)
from scripts.artifact_emotion_gated_attetnion_optimized.model import GatedBranch, score_or
from scripts.artifact_emotion_gated_attetnion_optimized.progress import Progress, status, duration


def new_directory(path):
    path.mkdir(parents=True, exist_ok=False)


def tensors(data):
    return (torch.as_tensor(data['artifact'],dtype=torch.float32),
            torch.as_tensor(data['emotion'],dtype=torch.float32),
            torch.as_tensor(data['emotion_valid'],dtype=torch.bool))


@torch.inference_mode()
def predict(model, data, device, batch_size=256, stage="PREDICT"):
    model.eval(); values = []; done = 0
    progress = Progress(stage, len(data["sample_id"]))
    for batch in DataLoader(TensorDataset(*tensors(data)), batch_size=batch_size, shuffle=False):
        logits, _ = model(*(x.to(device) for x in batch))
        values.append(torch.sigmoid(logits).cpu().numpy())
        done += len(batch[0]); progress.update(done, force=len(values) == 1)
    return np.concatenate(values)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('Invalid training hyperparameters')
    status(f'TRAIN {args.modality}', 'Loading and verifying train/dev feature caches...')
    train_data, tm = load_cache(args.train_cache,args.modality,'train')
    dev, dm = load_cache(args.dev_cache,args.modality,'dev')
    disjoint(train_data,dev)
    if tm['recipe_sha256'] != dm['recipe_sha256'] or tm['recipe']['implementation'] != implementation():
        raise ValueError('Train/dev extraction recipes differ or implementation changed')
    label_index = 1 if args.modality == 'video' else 2
    keep = train_data['labels'][:,label_index] >= 0
    dev_keep = dev['labels'][:,label_index] >= 0
    labels = train_data['labels'][keep,label_index]
    if len(set(labels)) != 2 or len(set(dev['labels'][dev_keep,label_index])) != 2:
        raise ValueError('Both classes required in train and dev for AUC selection')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = GatedBranch(tm['artifact_dim'],tm['emotion_dim']).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    weight = float((labels==0).sum()/(labels==1).sum()) if args.class_balance else 1.0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=args.device))
    a,e,v = tensors(train_data)
    loader = DataLoader(TensorDataset(a[keep],e[keep],v[keep],torch.tensor(labels,dtype=torch.float32)),
        batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed),num_workers=0)
    new_directory(args.output);best = -1.;history = []
    configuration = dict(modality=args.modality,artifact_dim=tm['artifact_dim'],emotion_dim=tm['emotion_dim'],
                         projection_dim=128,dropout=.2)
    started = time.monotonic()
    status(f'TRAIN {args.modality}', f'device={args.device} train={len(labels)} dev={int(dev_keep.sum())} '
           f'epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} output={args.output}')
    for epoch in range(args.epochs):
        epoch_stage = f'TRAIN {args.modality} epoch {epoch+1}/{args.epochs}'
        progress = Progress(epoch_stage, len(loader), unit='batches')
        model.train();total = 0.;count = 0
        for batch_index,(a,e,v,y) in enumerate(loader,1):
            optimizer.zero_grad(set_to_none=True)
            logits,_ = model(a.to(args.device),e.to(args.device),v.to(args.device))
            loss = criterion(logits,y.to(args.device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
            total+=float(loss.detach())*len(y);count+=len(y)
            progress.update(batch_index, f'samples={count}/{len(labels)} loss={total/count:.6f} '
                            f'lr={optimizer.param_groups[0]["lr"]:.6g}')
        probabilities = predict(model,dev,args.device,stage=f'DEV {args.modality} epoch {epoch+1}/{args.epochs}')
        report = metrics(dev['labels'][dev_keep,label_index],probabilities[dev_keep])
        history.append(dict(epoch=epoch,train_loss=total/count,dev=report))
        if report['roc_auc'] > best:
            best=report['roc_auc']
            state=dict(format='artifact_emotion_gated_head_v1',config=configuration,
                model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                recipe_sha256=tm['recipe_sha256'],recipe=tm['recipe'],implementation=implementation(),
                train_cache_sha256=tm['sha256'],dev_cache_sha256=dm['sha256'],
                seen_ids=sorted(set(train_data['sample_id'].tolist()+dev['sample_id'].tolist())),
                seen_groups=sorted(set(train_data['group'].tolist()+dev['group'].tolist())),
                best_epoch=epoch,dev_metrics=report,seed=args.seed,pos_weight=weight,
                training=dict(epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,weight_decay=1e-4,label_index=label_index))
            torch.save(state,args.output/'best.pt')
            status(epoch_stage, f'BEST checkpoint saved: dev_auc={best:.6f}')
        write(args.output/'history.json',history)
        elapsed = time.monotonic()-started
        remaining = elapsed/(epoch+1)*(args.epochs-epoch-1)
        status(epoch_stage, f'COMPLETE loss={total/count:.6f} dev_auc={report["roc_auc"]:.6f} '
               f'dev_ba={report["balanced_accuracy"]:.6f} best_auc={best:.6f} '
               f'train_elapsed={duration(elapsed)} remaining_epochs_ETA={duration(remaining)}')
    write(args.output/'COMPLETED.json',dict(best_dev_auc=best,checkpoint_sha256=sha(args.output/'best.pt'),
        training_rows=int(keep.sum()),unknown_training_labels=int((~keep).sum()),
        emotion_missing_train=int((~train_data['emotion_valid']).sum()),threshold=.5))
    status(f'TRAIN {args.modality}', f'COMPLETE best_dev_auc={best:.6f} checkpoint={args.output / "best.pt"}')


def load_head(path, device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['format']!='artifact_emotion_gated_head_v1' or state['implementation']!=implementation():
        raise ValueError('Unknown/stale gated checkpoint implementation')
    c=state['config']
    if c['projection_dim']!=128:raise ValueError('Expected 128-dimensional projection')
    model=GatedBranch(c['artifact_dim'],c['emotion_dim'],c['dropout']).to(device)
    model.load_state_dict(state['model'],strict=True);model.eval()
    return model,state


def lock(args):
    status('LOCK', 'Verifying video/audio best checkpoints...')
    if args.output.exists():raise FileExistsError('Lock already exists')
    branches={};data_provenance=[]
    for modality,path in [('video',args.video_head),('audio',args.audio_head)]:
        _,state=load_head(path)
        if state['config']['modality']!=modality:raise ValueError('Head modality mismatch')
        if not (path.parent/'COMPLETED.json').is_file() or read(path.parent/'COMPLETED.json')['checkpoint_sha256']!=sha(path):
            raise ValueError('Head training incomplete or checkpoint changed')
        recipe=state['recipe'];data_provenance.append(tuple(recipe[k] for k in ('dataset','registry_sha256','eligibility_sha256','manifest_sha256')))
        branches[modality]=dict(checkpoint=str(path.resolve()),sha256=sha(path),recipe_sha256=state['recipe_sha256'])
    if data_provenance[0]!=data_provenance[1]:raise ValueError('Modality data provenance differs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write(args.output,dict(format='gated_or_lock_v1',branches=branches,threshold=.5,
                          fusion='1-(1-Vs)*(1-As)',implementation=implementation()))
    status('LOCK', f'COMPLETE {args.output}')


def evaluate(args):
    status(f'EVALUATE {args.split}', 'Verifying lock, heads and feature caches...')
    frozen=read(args.lock)
    if frozen['format']!='gated_or_lock_v1' or frozen['implementation']!=implementation() or frozen['threshold']!=.5 or frozen['fusion']!='1-(1-Vs)*(1-As)':
        raise ValueError('Invalid/stale lock')
    data={};scores={};reports={}
    for modality,path in [('video',args.video_cache),('audio',args.audio_cache)]:
        branch=frozen['branches'][modality]
        if sha(branch['checkpoint'])!=branch['sha256']:raise ValueError('Locked head changed')
        model,state=load_head(branch['checkpoint'],args.device)
        d,m=load_cache(path,modality,args.split)
        if state['config']['modality']!=modality or m['recipe_sha256']!=branch['recipe_sha256']:
            raise ValueError('Evaluation recipe/modality differs from locked head')
        if args.split=='test':
            for key,seen in [('sample_id','seen_ids'),('group','seen_groups')]:
                if set(d[key])&set(state[seen]):raise ValueError('Test leakage in '+key)
        elif m['sha256']!=state['dev_cache_sha256']:
            raise ValueError('Unexpected selection-dev cache')
        data[modality]=d;scores[modality]=predict(model,d,args.device,stage=f'EVALUATE {args.split} {modality}')
        index=1 if modality=='video' else 2;keep=d['labels'][:,index]>=0
        reports[modality]=metrics(d['labels'][keep,index],scores[modality][keep])
    vi={sid:i for i,sid in enumerate(data['video']['sample_id'])};ai={sid:i for i,sid in enumerate(data['audio']['sample_id'])}
    rows=[];truth=[];final=[];baseline=[]
    for sid in sorted(set(vi)|set(ai)):
        v=vi.get(sid);a=ai.get(sid);row=dict(sample_id=str(sid),video_present=v is not None,audio_present=a is not None,
            video_label='',audio_label='',clip_label='',video_score='',audio_score='',fused_score='',artifact_or='',included=False)
        for mod,idx,labelidx in [('video',v,1),('audio',a,2)]:
            if idx is not None:
                row[mod+'_label']=int(data[mod]['labels'][idx,labelidx]);row[mod+'_score']=float(scores[mod][idx])
                row['clip_label']=int(data[mod]['labels'][idx,0])
        if v is not None and a is not None:
            if data['video']['group'][v]!=data['audio']['group'][a] or not np.array_equal(data['video']['labels'][v],data['audio']['labels'][a]):
                raise ValueError('Cross-modality ID/label/group mismatch')
            row['fused_score']=float(score_or(scores['video'][v],scores['audio'][a]))
            row['artifact_or']=float(score_or(data['video']['artifact_probability'][v],data['audio']['artifact_probability'][a]))
            if row['clip_label']>=0:
                row['included']=True;truth.append(row['clip_label']);final.append(row['fused_score']);baseline.append(row['artifact_or'])
        rows.append(row)
    reports['gated_or']=metrics(truth,final);reports['artifact_or_same_cohort']=metrics(truth,baseline)
    reports['coverage']=dict(video_rows=len(vi),audio_rows=len(ai),union_rows=len(rows),common_ids=len(set(vi)&set(ai)),
        fusion_evaluated=len(truth),missing_video=len(set(ai)-set(vi)),missing_audio=len(set(vi)-set(ai)),
        video_emotion_missing=int((~data['video']['emotion_valid']).sum()),audio_emotion_missing=int((~data['audio']['emotion_valid']).sum()))
    new_directory(args.output)
    with (args.output/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(args.output/'metrics.json',reports)
    write(args.output/'COMPLETED.json',dict(lock_sha256=sha(args.lock),split=args.split,
        video_cache_sha256=sha(args.video_cache/'features.npz'),audio_cache_sha256=sha(args.audio_cache/'features.npz')))
    print(json.dumps(reports,indent=2),flush=True)
    status(f'EVALUATE {args.split}', f'COMPLETE fused_samples={len(truth)} output={args.output}')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('preflight');p.add_argument('--trusted-artifact-checkpoints',action='store_true');p.add_argument('--output',type=Path)
    p=sub.add_parser('extract');p.add_argument('--modality',choices=['video','audio'],required=True);p.add_argument('--split',choices=['train','dev','test'],required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path);p.add_argument('--trusted-artifact-checkpoints',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--face-batch-size',type=int,default=1)
    p.add_argument('--reuse-legacy-tokens',type=Path);p.add_argument('--stop-after',type=int,default=0,help='Stop after N newly committed samples for recovery testing')
    p=sub.add_parser('train');p.add_argument('--modality',choices=['video','audio'],required=True)
    p.add_argument('--train-cache',type=Path,required=True);p.add_argument('--dev-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--class-balance',action='store_true',help='Train-only pos_weight; default unweighted BCE')
    p=sub.add_parser('lock');p.add_argument('--video-head',type=Path,required=True);p.add_argument('--audio-head',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('--video-cache',type=Path,required=True);p.add_argument('--audio-cache',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    args=parser.parse_args(argv)
    status(args.command.upper(), 'START')
    if args.command=='preflight':
        from scripts.artifact_emotion_gated_attetnion_optimized.features import dependencies
        results=[]
        for modality in ('video','audio'):
            status('PREFLIGHT', f'Checking {modality} checkpoint, local files, splits and labels...')
            try:results.append(dependencies(modality,args.trusted_artifact_checkpoints))
            except Exception as error:results.append(dict(modality=modality,issues=[str(error)]))
        print(json.dumps(results,indent=2),flush=True)
        for result in results:
            label = 'BLOCKED' if result['issues'] else 'PASS'
            status('PREFLIGHT', f"{result['modality']}: {label} ({len(result['issues'])} issues)")
        blocked = any(r['issues'] for r in results)
        status('PREFLIGHT', 'BLOCKED (exit code 2): training has NOT started; resolve the listed issues.' if blocked
               else 'PASS (exit code 0): prerequisite checks passed; this command does not train a model.')
        if args.output:
            if args.output.exists():raise FileExistsError('Preflight output exists')
            args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,results)
        return 2 if any(r['issues'] for r in results) else 0
    if args.command=='extract':
        from scripts.artifact_emotion_gated_attetnion_optimized.features import extract
        extract(args)
    else:{'train':train,'lock':lock,'evaluate':evaluate}[args.command](args)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
