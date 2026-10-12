"""FAV -> FAV frozen-artifact hierarchical temporal-emotion experiment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='scripts.hierarchical_score_fusion_GRU_Temproal_Attention'
import argparse
import csv
import time
import traceback
import numpy as np
import torch
from .common import ROOT,SOURCE,OUTPUT,CLASSES,require,read,sha,digest,atomic_json,implementation,metrics,score_or,WriterLock,status
from .data import catalog
from .extract import assets,Encoders,prepare
from .train import fit,load_head,predict


def evaluate(data,heads,output,lock,cfg):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    v,a=data['video'],data['audio']
    for k in ('sample_id','group','labels'):require(np.array_equal(v[k],a[k]),'Test modality alignment mismatch')
    ev,wv=predict(heads['video'],v,'video',cfg['head_device'],cfg['batch_size'],attention=True,progress=True)
    ea,wa=predict(heads['audio'],a,'audio',cfg['head_device'],cfg['batch_size'],attention=True,progress=True)
    validv=np.isfinite(ev);valida=np.isfinite(ea);complete=validv&valida
    av=v['artifact_probability'].astype(np.float64);aa=a['artifact_probability'].astype(np.float64)
    artifact=score_or(av,aa)
    # Missing observations are omitted from OR; they are reported as missing, never labeled real.
    emotion=score_or(np.where(validv,ev,0.),np.where(valida,ea,0.))
    final=score_or(artifact,emotion)
    require(np.all(final+1e-12>=artifact),'Hierarchical OR must not lower artifact probability')
    labels=v['labels'];known=labels[:,0]>=0
    def cohort(mask):return {k:metrics(labels[mask,0],s[mask]) for k,s in [('artifact_or',artifact),('emotion_or',emotion),('hierarchical_or',final)]}
    report=dict(source='FAV',target='FAV',split='test',smoke_only=cfg['smoke_only'],threshold=.5,
        primary_complete_emotion=cohort(known&complete),secondary_full_cohort=cohort(known),
        video_emotion=metrics(labels[validv&(labels[:,1]>=0),1],ev[validv&(labels[:,1]>=0)]),
        audio_emotion=metrics(labels[valida&(labels[:,2]>=0),2],ea[valida&(labels[:,2]>=0)]),
        coverage=dict(planned=len(labels),evaluated=int(known.sum()),complete_emotion=int(complete.sum()),
            video_emotion_missing=int((~validv).sum()),audio_emotion_missing=int((~valida).sum())),
        fusion='A=1-(1-Av)(1-Aa); E=1-(1-Ev)(1-Ea); Final=1-(1-A)(1-E)',
        missing_policy='secondary: omit unavailable emotion terms; primary: both available',
        fixed_artifact_scores=True,lock_sha256=sha(lock))
    temp=output/'predictions.csv.partial'
    names=['sample_id','group','clip_label','video_label','audio_label','Av','Aa','Ev','Ea','artifact_or','emotion_or','hierarchical_or','video_emotion_valid','audio_emotion_valid','included_primary']
    with temp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(names)
        for i in range(len(labels)):
            writer.writerow([str(v['sample_id'][i]),str(v['group'][i]),*labels[i].tolist(),float(av[i]),float(aa[i]),
                float(ev[i]) if validv[i] else '',float(ea[i]) if valida[i] else '',float(artifact[i]),float(emotion[i]),float(final[i]),bool(validv[i]),bool(valida[i]),bool(complete[i]&known[i])])
        f.flush();os.fsync(f.fileno())
    os.replace(temp,output/'predictions.csv')
    for mod,w in [('video',wv),('audio',wa)]:
        d=data[mod];np.savez_compressed(output/(mod+'_attention.npz'),sample_id=d['sample_id'],offsets=d['offsets'],
            timestamp=d['timestamp'],valid=d['valid'],attention=w)
    atomic_json(output/'metrics.json',report)
    atomic_json(output/'COMPLETED.json',dict(lock_sha256=sha(lock),files={n:sha(output/n) for n in
        ['metrics.json','predictions.csv','video_attention.npz','audio_attention.npz']}))
    status('EVALUATE',str(report));return report


def make_config(args):
    return dict(format='fav_hierarchical_temporal_v1',seed=args.seed,epochs=args.epochs,batch_size=args.batch_size,
        lr=args.lr,weight_decay=1e-4,pos_weight=args.pos_weight,encoder_device=args.encoder_device,head_device=args.head_device,
        threads=args.threads,frame_batch_size=args.frame_batch_size,smoke_limit=args.smoke_limit,smoke_only=bool(args.smoke_limit),
        video_input=dict(frames=128,crop=256,resize_short=256,region='X3D-identical center crop',
                         normalization='HSE ImageNet mean/std; no 224 resize; no face detector; duplicate indices retained'),
        audio_input='reuse exact 16k mono 3s/hop1s unpadded utterance token embeddings; short clip whole; native layer norm',
        head=dict(projection=128,projection_norm='BatchNorm1d over valid tokens only',dropout=.2,
                  video_hidden=64,video_layers=1,audio_hidden=128,audio_layers=2,bidirectional=False,
                  attention='softmax(linear(GRU hidden)), padding masked',native_classes=CLASSES),
        threshold=.5,fusion='artifact_OR_then_emotion_OR_then_final_OR')


def run(args):
    require(args.epochs>=1 and args.batch_size>=1 and args.lr>0 and args.pos_weight>0 and args.frame_batch_size>=1 and args.threads>=1,'Invalid settings')
    require(args.smoke_limit>=0,'Smoke limit cannot be negative')
    if args.smoke_limit:require(args.output.resolve()!=OUTPUT.resolve(),'Smoke requires a separate --output directory')
    for device in (args.encoder_device,args.head_device):
        if device.startswith('cuda'):require(torch.cuda.is_available(),'CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    status('PREFLIGHT','Checking completed FAV caches and fixed artifact checkpoint identities...')
    sources,source_identity=catalog(args.source_cache,args.smoke_limit,args.seed)
    asset=assets()
    for split in ('train','dev','test'):
        recipe=sources['audio'][split]['meta']['recipe']
        require(recipe['revision']==asset['seed_revision'] and recipe['emotion_pretrained']['files']['model.pt']==asset['seed_sha256'],'Audio token encoder differs from installed seed checkpoint')
    cfg=make_config(args);descriptor=dict(config=cfg,source=source_identity,assets=asset,implementation=implementation())
    identity=digest(descriptor)
    if args.command=='preflight':
        status('PREFLIGHT',f'PASS; code_id={identity}; source files/checkpoints verified; media stamps checked during preparation')
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with WriterLock(args.output):
        plan=args.output/'RUN.json'
        if plan.exists():
            require(args.resume,'Output exists; use --resume')
            require(read(plan)==descriptor,'Run inputs/settings/code changed; use another output folder')
        else:atomic_json(plan,descriptor)
        complete=args.output/'COMPLETED.json'
        if complete.exists():
            old=read(complete)
            require(old['identity']==identity,'Run identity mismatch')
            for path,h in old['files'].items():require(sha(args.output/path)==h,'Completed output changed: '+path)
            status('PIPELINE','Already complete; saved result verified');return
        encoders=Encoders(asset,args.encoder_device,args.frame_batch_size);prepared={m:{} for m in ('video','audio')}
        t=time.perf_counter();timings={}
        for mod in prepared:
            for split in ('train','dev'):
                cache_identity=dict(run=identity,modality=mod,split=split,source=source_identity[mod][split])
                prepared[mod][split]=prepare(sources[mod][split],mod,split,args.output/'cache'/mod/split,cache_identity,encoders,args.resume)
        timings['train_dev_prepare_seconds']=time.perf_counter()-t
        if args.prepare_only:
            atomic_json(args.output/'PREPARED_TRAIN_DEV.json',dict(identity=identity,timing=timings));status('PREPARE','Train/dev complete; rerun with --resume to train');return
        encoders.hse=None
        if torch.cuda.is_available():torch.cuda.empty_cache()
        head_ids={};selections={};t=time.perf_counter()
        for mod in prepared:
            head_ids[mod]=digest(dict(run=identity,modality=mod,train_cache=prepared[mod]['train']['completion_sha256'],dev_cache=prepared[mod]['dev']['completion_sha256']))
            selections[mod]=fit(prepared[mod]['train'],prepared[mod]['dev'],mod,args.output/'heads'/mod,cfg,head_ids[mod])
        timings['training_seconds']=time.perf_counter()-t
        locked=dict(identity=identity,heads={mod:dict(identity=head_ids[mod],checkpoint_sha256=sha(args.output/'heads'/mod/'best.pt')) for mod in prepared},
            source_selection='FAV dev modality AUC only',threshold=.5,fusion=cfg['fusion'],artifact_sources=source_identity)
        lock=args.output/'LOCKED_BEFORE_TEST.json'
        if lock.exists():require(read(lock)==locked,'Frozen test lock changed')
        else:atomic_json(lock,locked)
        # New test sequence extraction and head inference happen only after both heads are selected and locked.
        test={};t=time.perf_counter()
        for mod in prepared:
            cache_identity=dict(run=identity,modality=mod,split='test',source=source_identity[mod]['test'])
            test[mod]=prepare(sources[mod]['test'],mod,'test',args.output/'cache'/mod/'test',cache_identity,encoders,args.resume)
        timings['test_prepare_seconds']=time.perf_counter()-t
        heads={mod:load_head(args.output/'heads'/mod,mod,head_ids[mod],args.head_device) for mod in prepared}
        for mod in heads:require(sha(args.output/'heads'/mod/'best.pt')==locked['heads'][mod]['checkpoint_sha256'],'Head changed after lock')
        t=time.perf_counter();report=evaluate(test,heads,args.output/'test',lock,cfg);timings['evaluation_seconds']=time.perf_counter()-t
        atomic_json(args.output/'SUMMARY.json',dict(metrics=report,selection=selections,smoke_only=cfg['smoke_only']))
        atomic_json(args.output/'timing.json',dict(seconds_this_invocation=timings))
        files=['RUN.json','LOCKED_BEFORE_TEST.json','SUMMARY.json','timing.json','test/COMPLETED.json',
            'test/metrics.json','test/predictions.csv','test/video_attention.npz','test/audio_attention.npz',
            'heads/video/best.pt','heads/audio/best.pt']
        atomic_json(complete,dict(identity=identity,source='FAV',target='FAV',smoke_only=cfg['smoke_only'],files={n:sha(args.output/n) for n in files}))
        status('PIPELINE',f'COMPLETE smoke_only={cfg["smoke_only"]} output={args.output}')


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','run'])
    p.add_argument('--source-cache',type=Path,default=SOURCE);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--encoder-device',default='cuda:0');p.add_argument('--head-device',default='cuda:0')
    p.add_argument('--frame-batch-size',type=int,default=32);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=5e-4);p.add_argument('--pos-weight',type=float,default=.25)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--smoke-limit',type=int,default=0)
    p.add_argument('--resume',action='store_true');p.add_argument('--prepare-only',action='store_true')
    return p

if __name__=='__main__':
    args=parser().parse_args()
    try:run(args)
    except KeyboardInterrupt:
        status('INTERRUPTED','Committed samples/epochs retained. Rerun the SAME command with --resume.');sys.exit(130)
    except Exception:
        traceback.print_exc();status('FAILED','No successful completion was recorded for the failed stage.');sys.exit(1)
