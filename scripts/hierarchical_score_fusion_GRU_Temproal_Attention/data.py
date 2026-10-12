"""Read-only import of completed FAV artifacts, and packed temporal datasets."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import ROOT,ARTIFACTS,DIMS,CLASSES,sha,digest,read,require,status
from scripts.artifact_emotion_gated_attetnion_optimized.common import load_cache
from scripts.artifact_emotion_gated_attetnion_optimized.local_inputs import local_rows


def subset_indices(data,limit,seed):
    if not limit:return np.arange(len(data['sample_id']))
    require(limit>=4,'Smoke limit must be at least 4')
    rng=np.random.default_rng(seed); buckets=[]
    for pair in sorted(set(map(tuple,data['labels'][:,1:].tolist()))):
        ids=np.flatnonzero((data['labels'][:,1:]==pair).all(1));rng.shuffle(ids);buckets.append(ids.tolist())
    picked=[]
    while any(buckets) and len(picked)<min(limit,len(data['sample_id'])):
        for b in buckets:
            if b and len(picked)<limit:picked.append(b.pop())
    return np.asarray(sorted(picked))


def catalog(source,limit=0,seed=42):
    source=Path(source); result={m:{} for m in DIMS};identities={m:{} for m in DIMS}
    reg=ROOT/'configs/fixed_manifests.json';quality=ROOT/'reports/phase0_v1/eligibility_v4.jsonl'
    manifest=ROOT/read(reg)['datasets']['FAV']['manifest']
    expected={'registry_sha256':sha(reg),'eligibility_sha256':sha(quality),'manifest_sha256':sha(manifest)}
    for mod in DIMS:
        checkpoint=ARTIFACTS/('x3d' if mod=='video' else 'aasist')/'best.pt';checkpoint_hash=sha(checkpoint)
        current={r['sample_id']:r for r in local_rows(reg,quality,mod)}
        for split_no,split in enumerate(('train','dev','test')):
            folder=source/mod/split; data,meta=load_cache(folder,mod,split)
            recipe=meta['recipe'];require(recipe['dataset']=='FAV','Only FAV caches may be used')
            for key,value in expected.items():require(recipe[key]==value,'Frozen registry/manifest/eligibility changed: '+key)
            require(recipe['artifact_checkpoint_sha256']==checkpoint_hash,'Artifact weights differ from completed source run')
            if mod=='video':
                require(recipe['artifact_input']==dict(num_frames=128,sampling='uniform_over_full_video',resize_short_side=256,crop_size=256,
                    normalization=dict(mean=[.45]*3,std=[.225]*3)),'Unexpected frozen X3D input')
            else:
                require(recipe['artifact_input']==dict(sample_rate=16000,channels='mono',length_policy='pad_to_eligible_train_max',
                    target_num_samples=284672,truncation='prohibited',padding='right_zero_pad'),'Unexpected frozen AASIST input')
            full_ids=set(data['sample_id'].tolist())
            require(full_ids=={sid for sid,r in current.items() if r['split']==split},'Source cache cohort differs from fixed manifest')
            selected=subset_indices(data,limit,seed+split_no)
            small={k:data[k][selected].copy() for k in ('sample_id','group','labels','artifact_probability')}
            rows=[current[str(sid)] for sid in small['sample_id']]
            for i,r in enumerate(rows):
                require(str(small['group'][i])==r['group'] and small['labels'][i].tolist()==r['labels'],'ID/group/label mismatch')
            identity=dict(source_folder=str(folder.resolve()),features_sha256=meta['sha256'],manifest_sha256=sha(folder/'manifest.json'),
                recipe_sha256=meta['recipe_sha256'],artifact_checkpoint_sha256=checkpoint_hash,
                ids_sha256=digest(small['sample_id'].tolist()),n=len(rows))
            result[mod][split]=dict(data=small,rows=rows,folder=folder,meta=meta,identity=identity)
            identities[mod][split]=identity
            status('PREFLIGHT',f'{mod} {split}: {len(rows)} / {meta["n"]} completed source rows')
        for a,b in (('train','dev'),('train','test'),('dev','test')):
            for k in ('sample_id','group'):
                require(not(set(result[mod][a]['data'][k])&set(result[mod][b]['data'][k])),'Split overlap: '+k)
    for split in ('train','dev','test'):
        for key in ('sample_id','group','labels'):
            require(np.array_equal(result['video'][split]['data'][key],result['audio'][split]['data'][key]),'Modality alignment mismatch: '+key)
    return result,identities


def check_media(source,row):
    p=Path(row['path']);require(p.is_file() and p.stat().st_size>0,'Missing/empty media: '+str(p))
    stat=p.stat(); marker=source['folder']/'samples'/(digest(row['sample_id'])+'.json')
    old=read(marker)
    require((stat.st_size,stat.st_mtime_ns)==(old['size'],old['mtime_ns']),'Media changed since artifact extraction: '+str(p))
    return [stat.st_size,stat.st_mtime_ns]


def load_packed(folder,identity):
    folder=Path(folder);m=read(folder/'COMPLETED.json');require(m['identity']==identity,'Temporal cache identity changed')
    for name,expected in m['files'].items():require(sha(folder/name)==expected,'Packed cache checksum mismatch: '+name)
    with np.load(folder/'index.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    for key in ('embedding','probabilities','timestamp','valid'):data[key]=np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
    data['folder']=folder;data['completion_sha256']=sha(folder/'COMPLETED.json')
    require(len(data['sample_id'])==m['n'] and len(set(data['sample_id']))==m['n'],'Invalid packed IDs')
    require(data['offsets'][0]==0 and data['offsets'][-1]==len(data['valid']) and np.all(np.diff(data['offsets'])>0),'Invalid offsets')
    return data


class SequenceDataset(Dataset):
    def __init__(self,data,modality,labeled=False):
        self.data=data;self.label_index=1 if modality=='video' else 2
        keep=data['valid_counts']>0
        if labeled:keep=keep&(data['labels'][:,self.label_index]>=0)
        self.indices=np.flatnonzero(keep)
    def __len__(self):return len(self.indices)
    def __getitem__(self,j):
        i=int(self.indices[j]);d=self.data;a,b=map(int,d['offsets'][i:i+2]);valid=np.flatnonzero(d['valid'][a:b])+a
        return dict(index=i,positions=valid,embedding=torch.from_numpy(np.array(d['embedding'][valid],copy=True)),
            probabilities=torch.from_numpy(np.array(d['probabilities'][valid],copy=True)),label=float(d['labels'][i,self.label_index]))


def collate(items):
    lengths=[len(r['embedding']) for r in items];t=max(lengths);b=len(items)
    e=torch.zeros(b,t,items[0]['embedding'].shape[1]);p=torch.zeros(b,t,items[0]['probabilities'].shape[1]);mask=torch.zeros(b,t,dtype=torch.bool)
    for i,r in enumerate(items):n=lengths[i];e[i,:n]=r['embedding'];p[i,:n]=r['probabilities'];mask[i,:n]=True
    return e,p,mask,torch.tensor([r['label'] for r in items]),[r['index'] for r in items],[r['positions'] for r in items]
