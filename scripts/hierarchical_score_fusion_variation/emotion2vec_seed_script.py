"""Frozen emotion trajectories -> local changes -> supervised heads -> score OR."""
from pathlib import Path
import argparse, json, sys, os, subprocess, random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.hierarchical_score_fusion.hierarchical_score_fusion_scripts import (
    load_cache as load_artifact_cache, no_overlap, metrics, write_json, digest, Logger, atomic_torch_save)
from src.models.hierarchical_score_fusion.hierarchical_checkpoint_loading import sha256
from src.branches.hierarchical_score_fusion_variation.hierarchical_emotion_variation_branch import HierarchicalEmotionVariationBranch
from src.datasets.hierarchical_score_fusion_variation.variation_features import change_features
from src.fusion.hierarchical_score_fusion_variation.hierarchical_variation_fusion import fuse


from src.models.hierarchical_score_fusion_variation.artifact_provenance import artifact_recipe_digest


def code_hashes():
    paths=sorted(p for base in ('src','scripts') for p in (ROOT/base).rglob('*.py')
                 if 'hierarchical_score_fusion_variation' in p.parts)
    paths += [ROOT/'src/models/hierarchical_score_fusion/emotion_deepfake_head.py',
              ROOT/'src/fusion/hierarchical_score_fusion/hierarchical_score_fusion.py',
              ROOT/'scripts/hierarchical_score_fusion/hierarchical_score_fusion_scripts.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def load_cache(path,split):
    m=json.loads((path/'manifest.json').read_text())
    if m['format']!='variation_seed_cache_v3' or m['split']!=split:raise ValueError('Wrong variation cache/split')
    if sha256(path/'cache.npz')!=m['cache_sha256'] or digest(m['recipe'])!=m['recipe_sha256']:raise ValueError('Cache hash mismatch')
    with np.load(path/'cache.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    n=m['n']
    if not n or any(len(v)!=n for v in d.values()) or len(set(d['sample_id']))!=n:raise ValueError('Invalid cache rows')
    if d['labels'].shape!=(n,3) or not np.isin(d['labels'],[-1,0,1]).all():raise ValueError('Invalid labels')
    for name in ['artifact_video','artifact_audio']:
        if d[name].shape!=(n,) or not np.isfinite(d[name]).all() or ((d[name]<0)|(d[name]>1)).any():raise ValueError('Invalid artifact scores')
    for mod,dim in [('video',27),('audio',30)]:
        if d[mod+'_variation'].shape!=(n,dim) or not np.isfinite(d[mod+'_variation']).all():raise ValueError('Invalid variation')
        if not np.issubdtype(d[mod+'_count'].dtype,np.integer) or (d[mod+'_count']<0).any():raise ValueError('Invalid transition count')
    return d,m


def extract(args,log):
    import cv2
    from src.models.hsemotion import EmotionExtractor
    from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import FrozenEmotion2VecSeed
    import torchaudio
    from src.datasets.frozen import FrozenMediaDataset
    from src.datasets.media import MediaConfig,read_audio
    from src.datasets.hierarchical_score_fusion_variation.extract_trajectories import visual,auditory_seed
    old,manifest=load_artifact_cache(args.artifact_cache,args.split)
    if manifest.get('dataset')!='FAV':raise ValueError('Only FakeAVCeleb in-domain is enabled')
    if not 0<=args.shard_index<args.shards:raise ValueError('Invalid shard')
    ids=old['sample_id'][args.shard_index::args.shards].tolist()
    if not ids:raise ValueError('Empty shard')
    cfg=json.loads(args.config.read_text())
    if cfg.get('window_seconds',3)!=3 or cfg.get('hop_seconds',1)!=1:raise ValueError('This recipe requires 3s windows and 1s hop')
    resolve=lambda p:Path(p) if Path(p).is_absolute() else ROOT/p
    registry=args.registry or ROOT/'configs/fixed_manifests.json'
    for key,path in [('registry_sha256',registry),('eligibility_sha256',args.eligibility)]:
        if sha256(path)!=manifest['recipe'][key]:raise ValueError('Artifact/trajectory data provenance mismatch')
    ds=FrozenMediaDataset('FAV',args.split,MediaConfig(),sample_ids=ids,eligibility_path=args.eligibility,task='paired',registry=registry)
    rows={r['sample_id']:r for r in ds.rows}; index={s:i for i,s in enumerate(old['sample_id'])}
    hse=EmotionExtractor(device=args.device)
    hse.recognizer.model.eval()
    for param in hse.recognizer.model.parameters():param.requires_grad_(False)
    if cfg.get('audio_model')!='emotion2vec_plus_seed':raise ValueError('Expected emotion2vec+ seed config')
    audio_model=FrozenEmotion2VecSeed(resolve(cfg['seed_snapshot']),device=args.device)
    detector_path=Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'
    detector=cv2.CascadeClassifier(str(detector_path))
    if detector.empty():raise ValueError('Face detector missing')
    weight=Path.home()/'.hsemotion/enet_b0_8_best_afew.pt'
    recipe=dict(artifact_recipe_sha256=artifact_recipe_digest(manifest['recipe']),implementation=code_hashes(),
        wrappers={'hsemotion.py':sha256(ROOT/'src/models/hsemotion.py')},
        hse_weight=sha256(weight),audio_pretrained=audio_model.provenance,
        audio_class_names=audio_model.class_names,video_class_names=hse.class_names,
        pretraining_provenance='official frozen emotion2vec+ seed; no FAV finetuning',
        detector=sha256(detector_path),media_source_sha256=sha256(ROOT/'src/datasets/media.py'),
        video='128 uniform unique frames; 8 probabilities; Haar single face; IoU .2 segments',
        audio='16k mono; complete 48000 sample windows; hop16000; native emotion2vec frontend; 9 probabilities including other/unknown',
        variation='video8/audio9 probability derivatives + TV rate; mean/std/p95abs; max gap1.5',
        preprocessing_provenance='native frozen emotion recognizer preprocessing',
        torchaudio=str(torchaudio.__version__),torch=str(torch.__version__),numpy=np.__version__,opencv=cv2.__version__)
    result={k:[] for k in ['sample_id','group','labels','artifact_video','artifact_audio','video_variation','audio_variation','video_count','audio_count']}
    log('extract_start',samples=len(ids),device=args.device,shard=args.shard_index)
    (args.output/'trajectories').mkdir()
    for position,sid in enumerate(ids,1):
        row=rows[sid];st=Path(row['path']).stat();i=index[sid]
        if st.st_size!=row['size'] or st.st_mtime_ns!=row['mtime_ns']:raise ValueError('Media changed after audit')
        if [-1 if x is None else x for x in row['labels']]!=old['labels'][i].tolist():raise ValueError('Label mismatch')
        with torch.inference_mode():
            v=visual(row['path'],hse,detector)
            waveform,has_audio=read_audio(row['path'],MediaConfig(sample_rate=16000,audio_target_num_samples=None))
            if not has_audio:raise ValueError('Missing audited audio')
            a=auditory_seed(waveform,audio_model)
        changes={mod:change_features(t['values'],t['times'],t['valid'],kind=mod,segments=t.get('segments')) for mod,t in [('video',v),('audio',a)]}
        payload={f'{mod}_{key}':value for mod,t in [('video',v),('audio',a)] for key,value in t.items()}
        payload.update({f'{mod}_{key}':value for mod,t in changes.items() for key,value in t.items()})
        payload['sample_id']=np.asarray(sid)
        np.savez_compressed(args.output/'trajectories'/f'{digest(sid)}.npz',**payload)
        for k in ['sample_id','group','labels','artifact_video','artifact_audio']:result[k].append(old[k][i])
        for mod in ['video','audio']:
            result[mod+'_variation'].append(changes[mod]['features'])
            result[mod+'_count'].append(changes[mod]['valid_pairs'])
        log('extract_sample',current=position,total=len(ids),sample_id=sid,video_pairs=changes['video']['valid_pairs'],audio_pairs=changes['audio']['valid_pairs'],video_exclusion=None if changes['video']['valid_pairs'] else 'no_valid_adjacent_pairs',audio_exclusion=None if changes['audio']['valid_pairs'] else 'fewer_than_two_complete_valid_windows')
    np.savez_compressed(args.output/'cache.npz',**{k:np.asarray(v) for k,v in result.items()})
    write_json(args.output/'manifest.json',dict(format='variation_seed_cache_v3',split=args.split,n=len(ids),recipe=recipe,recipe_sha256=digest(recipe),cache_sha256=sha256(args.output/'cache.npz'),artifact_cache_sha256=manifest['cache_sha256'],original_artifact_recipe_sha256=manifest['recipe_sha256']))


def extract_multi(args,log):
    old,_=load_artifact_cache(args.artifact_cache,args.split)
    if len(set(args.gpus))!=len(args.gpus) or len(old['sample_id'])<len(args.gpus):raise ValueError('Invalid GPU partition')
    children=[];handles=[]
    try:
        for rank,gpu in enumerate(args.gpus):
            dest=args.output/f'shard_{rank}'
            command=[sys.executable,str(Path(__file__).resolve()),'extract','--config',str(args.config),'--artifact-cache',str(args.artifact_cache),'--eligibility',str(args.eligibility),'--split',args.split,'--device','cuda:0','--output',str(dest),'--shards',str(len(args.gpus)),'--shard-index',str(rank)]
            if args.registry:command+=['--registry',str(args.registry)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            handle=(args.output/f'gpu_{gpu}.log').open('w');handles.append(handle)
            children.append(subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT))
        import time
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Extraction worker failed; inspect GPU logs')
            log('workers_running',completed=sum(p.poll()==0 for p in children),total=len(children))
            time.sleep(10)
        if any(p.returncode!=0 for p in children):raise RuntimeError('Extraction failed')
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in handles:h.close()
    shards=[load_cache(args.output/f'shard_{r}',args.split) for r in range(len(args.gpus))]
    if len({m['recipe_sha256'] for _,m in shards})!=1:raise ValueError('Shard recipes differ')
    merged={k:np.concatenate([d[k] for d,m in shards]) for k in shards[0][0]}
    ids=merged['sample_id'];lookup={s:i for i,s in enumerate(ids)}
    if len(lookup)!=len(ids) or set(ids)!=set(old['sample_id']):raise ValueError('Shard coverage mismatch')
    order=[lookup[s] for s in old['sample_id']]
    np.savez_compressed(args.output/'cache.npz',**{k:v[order] for k,v in merged.items()})
    m=dict(shards[0][1]);m.update(n=len(ids),cache_sha256=sha256(args.output/'cache.npz'))
    write_json(args.output/'manifest.json',m)


def evaluate_arrays(data,branch,output,split,device,log):
    from src.fusion.hierarchical_score_fusion_variation.analysis_report import export_analysis
    mask=(data['video_count']>0)&(data['audio_count']>0)&(data['labels'][:,0]>=0)
    if not mask.any():raise ValueError('No common valid clips')
    branch.eval();scores={k:[] for k in ['artifact_only','emotion_only','hierarchical','video_emotion','audio_emotion']}
    indices=np.flatnonzero(mask)
    with torch.no_grad():
        for start in range(0,len(indices),256):
            idx=indices[start:start+256]
            pred=branch(torch.tensor(data['video_variation'][idx],device=device,dtype=torch.float32),torch.tensor(data['audio_variation'][idx],device=device,dtype=torch.float32))
            out=fuse(torch.tensor(data['artifact_video'][idx],device=device),torch.tensor(data['artifact_audio'][idx],device=device),pred['video_score'],pred['audio_score'])
            for k in ['artifact_only','emotion_only','hierarchical']:scores[k].extend(out[k].cpu().tolist())
            scores['video_emotion'].extend(pred['video_score'].cpu().tolist())
            scores['audio_emotion'].extend(pred['audio_score'].cpu().tolist())
    y=data['labels'][mask,0]
    report={k:metrics(y,np.asarray(scores[k])) for k in ['artifact_only','emotion_only','hierarchical']}
    for mod,ix in [('video',1),('audio',2)]:
        labels=data['labels'][mask,ix];valid=labels>=0
        if valid.any():report[mod+'_emotion_common']=metrics(labels[valid],np.asarray(scores[mod+'_emotion'])[valid])
    # Also report each modality on its own full eligible set, avoiding paired-only selection bias.
    with torch.no_grad():
        for mod,ix in [('video',1),('audio',2)]:
            valid=(data[mod+'_count']>0)&(data['labels'][:,ix]>=0)
            if valid.any():
                x=torch.tensor(data[mod+'_variation'][valid],device=device,dtype=torch.float32)
                p=np.concatenate([getattr(branch,mod+'_head')(batch).sigmoid().cpu().numpy() for batch in x.split(256)])
                report[mod+'_emotion_all_valid']=metrics(data['labels'][valid,ix],p)
    valid=data['labels'][:,0]>=0
    artifact=1-(1-data['artifact_video'])*(1-data['artifact_audio'])
    report['artifact_only_all']=metrics(data['labels'][valid,0],artifact[valid])
    report['coverage']=dict(total=len(mask),common_valid=int(mask.sum()),excluded=int((~mask).sum()),video_no_pairs=int((data['video_count']==0).sum()),audio_no_pairs=int((data['audio_count']==0).sum()),
        excluded_by_clip_label={str(label):int(((~mask)&(data['labels'][:,0]==label)).sum()) for label in [-1,0,1]})
    report['or_changes_at_05']=export_analysis(data,scores,mask,report,output,split)
    report['by_modality_labels']={}
    for vl in (0,1):
        for al in (0,1):
            subgroup=(data['labels'][mask,1]==vl)&(data['labels'][mask,2]==al)
            if subgroup.any():report['by_modality_labels'][f'video{vl}_audio{al}']={k:metrics(y[subgroup],np.asarray(scores[k])[subgroup]) for k in ['artifact_only','emotion_only','hierarchical']}
    write_json(output/f'{split}_metrics.json',report)
    np.savez_compressed(output/f'{split}_predictions.npz',sample_id=data['sample_id'][mask],labels=y,**scores)
    log('evaluation_complete',split=split,metrics=report)


def train(args, log):
    tr, tm = load_cache(args.train_cache, 'train')
    dv, dm = load_cache(args.dev_cache, 'dev')
    if tm['recipe'].get('implementation',code_hashes()) != code_hashes():
        raise ValueError('Extraction implementation changed; inspect provenance before proceeding')
    if tm['recipe_sha256'] != dm['recipe_sha256']:
        raise ValueError('Train/dev feature recipe mismatch')
    no_overlap(tr, dv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device).startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    branch = HierarchicalEmotionVariationBranch(audio_dim=30).to(args.device)
    selection = {}
    for modality, index in [('video', 1), ('audio', 2)]:
        head = getattr(branch, modality+'_head')
        masks = [(d[modality+'_count']>0)&(d['labels'][:,index]>=0) for d in (tr,dv)]
        tx = torch.from_numpy(tr[modality+'_variation'][masks[0]]).float()
        ty = torch.from_numpy(tr['labels'][masks[0],index]).float()
        dx = torch.from_numpy(dv[modality+'_variation'][masks[1]]).float()
        dy = dv['labels'][masks[1],index]
        if len(torch.unique(ty)) != 2 or len(np.unique(dy)) != 2:
            raise ValueError(f'{modality}: both classes needed in train and dev')
        # Clip-level statistics fitted ONLY on valid, labeled train samples.
        head.fit_standardizer(tx.to(args.device))
        pw = float((ty==0).sum()/(ty==1).sum())
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=args.device))
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),
                    batch_size=args.batch_size, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(args.seed))
        best_auc, best_state, best_epoch, stale = -1., None, 0, 0
        log('head_train_start', modality=modality, train_n=len(tx), dev_n=len(dx), pos_weight=pw)
        for epoch in range(1, args.epochs+1):
            head.train()
            loss_sum, seen = 0., 0
            for batch_index, (x,y) in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(x.to(args.device)), y.to(args.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise FloatingPointError('Missing/nonfinite head gradient')
                optimizer.step()
                loss_sum += float(loss.detach())*len(x)
                seen += len(x)
                if batch_index % 50 == 0 or batch_index == len(loader):
                    log('train_batch', modality=modality, epoch=epoch, batch=batch_index,
                        batches=len(loader), train_loss=loss_sum/seen)
            head.eval()
            with torch.no_grad():
                probs = np.concatenate([head(x.to(args.device)).sigmoid().cpu().numpy() for x in dx.split(256)])
            auc = float(roc_auc_score(dy, probs))
            if auc > best_auc:
                best_auc, best_epoch, stale = auc, epoch, 0
                best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                atomic_torch_save(args.output/f'{modality}_head_best.pt',
                                  dict(state_dict=best_state, epoch=epoch, dev_auc=auc))
            else:
                stale += 1
            log('epoch_complete', modality=modality, epoch=epoch, train_loss=loss_sum/seen,
                dev_auc=auc, best_dev_auc=best_auc, best_epoch=best_epoch, patience_used=stale)
            if stale >= args.patience:
                break
        head.load_state_dict(best_state)
        selection[modality] = dict(best_epoch=best_epoch, best_dev_auc=best_auc,
                                   pos_weight=pw, train_n=len(tx), dev_n=len(dx))
    state = dict(format='variation_seed_heads_v3', hidden_dim=64, dropout=.2,
                 state_dict={k:v.detach().cpu() for k,v in branch.state_dict().items()},
                 recipe_sha256=tm['recipe_sha256'], recipe=tm['recipe'],
                 train_cache_sha256=tm['cache_sha256'], dev_cache_sha256=dm['cache_sha256'],
                 train_ids=tr['sample_id'].tolist(), dev_ids=dv['sample_id'].tolist(),
                 train_groups=tr['group'].tolist(), dev_groups=dv['group'].tolist(),
                 selection=selection, seed=args.seed, implementation=code_hashes())
    atomic_torch_save(args.output/'emotion_heads.pt', state)
    evaluate_arrays(dv, branch, args.output, 'dev', args.device, log)
    write_json(args.output/'training_summary.json', dict(selection=selection, seed=args.seed,
           emotion_checkpoint_sha256=sha256(args.output/'emotion_heads.pt'),
           epochs_limit=args.epochs, patience=args.patience, batch_size=args.batch_size,
           learning_rate=.001, weight_decay=.0001, train_cache_sha256=tm['cache_sha256'],
           dev_cache_sha256=dm['cache_sha256'], recipe_sha256=tm['recipe_sha256']))



def evaluate(args,log):
    d,m=load_cache(args.cache,args.split)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state['format']!='variation_seed_heads_v3' or state['recipe_sha256']!=m['recipe_sha256'] or state['implementation']!=code_hashes():raise ValueError('Checkpoint provenance mismatch')
    no_overlap(d,dict(sample_id=state['train_ids'],group=state['train_groups']))
    if args.split=='dev' and m['cache_sha256']!=state['dev_cache_sha256']:raise ValueError('Wrong selection dev cache')
    if args.split=='test':no_overlap(d,dict(sample_id=state['dev_ids'],group=state['dev_groups']))
    b=HierarchicalEmotionVariationBranch(audio_dim=30).to(args.device);b.load_state_dict(state['state_dict'])
    evaluate_arrays(d,b,args.output,args.split,args.device,log)




def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ['extract','extract-multi']:
        c=sub.add_parser(name)
        c.add_argument('--config',type=Path,required=True);c.add_argument('--artifact-cache',type=Path,required=True)
        c.add_argument('--eligibility',type=Path,required=True);c.add_argument('--registry',type=Path)
        c.add_argument('--split',choices=['train','dev','test'],required=True)
        if name=='extract':
            c.add_argument('--shards',type=int,default=1);c.add_argument('--shard-index',type=int,default=0)
        else:c.add_argument('--gpus',nargs='+',default=['0','1','2','3'])
    t=sub.add_parser('train');t.add_argument('--train-cache',type=Path,required=True);t.add_argument('--dev-cache',type=Path,required=True)
    t.add_argument('--seed',type=int,default=42);t.add_argument('--epochs',type=int,default=30);t.add_argument('--patience',type=int,default=5);t.add_argument('--batch-size',type=int,default=64)
    e=sub.add_parser('evaluate');e.add_argument('--cache',type=Path,required=True);e.add_argument('--checkpoint',type=Path,required=True);e.add_argument('--split',choices=['dev','test'],required=True)
    for c in [*sub.choices.values()]:
        c.add_argument('--output',type=Path,required=True);c.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory; existing results are preserved')
    args.output.mkdir(parents=True);torch.set_num_threads(2)
    log=Logger(args.output)
    try:
        {'extract':extract,'extract-multi':extract_multi,'train':train,'evaluate':evaluate}[args.command](args,log)
        write_json(args.output/'COMPLETED.json',dict(command=args.command))
    except BaseException as error:
        write_json(args.output/'FAILED.json',dict(error=repr(error)))
        import traceback
        (args.output/'error.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
