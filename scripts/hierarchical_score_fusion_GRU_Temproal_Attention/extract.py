"""Resumable 256px HSE extraction, audio-token reuse, and contiguous sequence caches."""
from pathlib import Path
import io
import hashlib
import sqlite3
import json
import time
import numpy as np
import torch
from .common import DIMS,CLASSES,require,read,sha,digest,atomic_json,Progress,status,WriterLock
from .data import check_media,load_packed


def assets():
    from src.models.hsemotion import hsemotion_checkpoint,HSE_SHA256
    from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import download_seed
    hse=hsemotion_checkpoint();require(sha(hse)==HSE_SHA256,'Official HSE asset changed')
    snapshot,revision=download_seed()
    return dict(hse_path=str(hse.resolve()),hse_sha256=HSE_SHA256,seed_path=str(Path(snapshot).resolve()),
                seed_revision=revision,seed_sha256=sha(Path(snapshot)/'model.pt'))


class Encoders:
    def __init__(self,asset,device,frame_batch):
        self.asset=asset;self.device=device;self.frame_batch=frame_batch;self.hse=None;self.audio_projection=None
    def load_video(self):
        if self.hse is None:
            from src.models.hsemotion import LocalHSEmotionRecognizer
            self.hse=LocalHSEmotionRecognizer(self.asset['hse_path'],self.device)
            self.hse.model.eval().requires_grad_(False)
            self.vw=torch.from_numpy(self.hse.classifier_weights).to(self.device)
            self.vb=torch.from_numpy(self.hse.classifier_bias).to(self.device)
    def load_audio(self):
        if self.audio_projection is None:
            state=torch.load(Path(self.asset['seed_path'])/'model.pt',map_location='cpu',weights_only=True)['model']
            self.audio_projection=(state['proj.weight'].clone(),state['proj.bias'].clone())
            names=(Path(self.asset['seed_path'])/'tokens.txt').read_text(encoding='utf-8').splitlines()
            names=[n.split('/')[-1].strip().lower() for n in names]
            names=[{'<unk>':'unknown','disgust':'disgusted','fear':'fearful','surprise':'surprised'}.get(n,n) for n in names]
            require(len(names)==9 and set(names)==set(CLASSES['audio']),'Seed class vocabulary changed')
            self.audio_order=[names.index(n) for n in CLASSES['audio']]
    @torch.inference_mode()
    def video(self,path):
        from src.datasets.media import read_video,MediaConfig
        self.load_video()
        # Same sampling and RGB center-crop pixels as X3D, with native HSE channel normalization.
        d=read_video(path,MediaConfig(frames=128,crop_size=256,resize_short=256,mean=(0.,)*3,std=(1.,)*3))
        frames=d['video'].permute(1,0,2,3)
        ids=d['frame_indices'].numpy();unique,first,inverse=np.unique(ids,return_index=True,return_inverse=True)
        frames=frames[first];out=[]
        mean=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
        for start in range(0,len(frames),self.frame_batch):
            x=frames[start:start+self.frame_batch].to(self.device)
            require(tuple(x.shape[1:])==(3,256,256),'HSE must receive 256x256 RGB tensors')
            out.append(self.hse.model((x-mean)/std).reshape(len(x),1280))
        embedding=torch.cat(out);prob=torch.softmax(torch.nn.functional.linear(embedding,self.vw,self.vb),dim=-1)
        return dict(embedding=embedding.cpu().numpy()[inverse],probabilities=prob.cpu().numpy()[inverse],
                    timestamp=d['timestamps'].numpy(),valid=np.ones(128,bool),frame_index=ids)
    @torch.inference_mode()
    def audio(self,source,row):
        self.load_audio();name=digest(row['sample_id'])+'.npz';path=source['folder']/'tokens'/name
        require(sha(path)==source['meta']['token_sha256'][name],'Completed audio token checksum mismatch')
        with np.load(path,allow_pickle=False) as z:
            require(str(z['sample_id'])==row['sample_id'],'Audio token ID mismatch')
            result={k:z[k].copy() for k in ('embedding','timestamp','valid','start_sample','end_sample')}
        e=torch.from_numpy(result['embedding']);w,b=self.audio_projection
        result['probabilities']=torch.softmax(torch.nn.functional.linear(e,w,b),dim=-1).numpy()[:,self.audio_order]
        return result


def validate_tokens(tokens,mod):
    e,p,t,m=[tokens[k] for k in ('embedding','probabilities','timestamp','valid')];n=len(t)
    require(n>0 and e.shape==(n,DIMS[mod]) and p.shape==(n,len(CLASSES[mod])) and m.shape==(n,) and m.dtype==np.bool_,'Invalid token dimensions')
    require(np.isfinite(t).all() and (np.diff(t)>=0).all(),'Nonmonotonic token timeline')
    require(np.isfinite(e[m]).all() and np.isfinite(p[m]).all() and ((p[m]>=0)&(p[m]<=1)).all(),'Nonfinite token')
    require(np.allclose(p[m].sum(1),1.,atol=1e-5),'Invalid token probabilities')
    if mod=='video':require(n==128 and m.all(),'HSE center-crop sequence must have 128 valid tokens')


def payload_decode(blob,checksum):
    require(hashlib.sha256(blob).hexdigest()==checksum,'Resumable row payload checksum mismatch')
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def prepare(source,mod,split,folder,identity,encoders,resume=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with WriterLock(folder):
        started=folder/'STARTED.json'
        if started.exists():
            require(resume,'Output exists; use --resume')
            require(read(started)==identity,'Extraction settings/source/code changed; use a new output')
        else:atomic_json(started,identity)
        for row in source['rows']:check_media(source,row)
        if (folder/'COMPLETED.json').exists():
            result=load_packed(folder,identity);status('CACHE',f'{mod} {split}: complete cache verified');return result
        db=sqlite3.connect(folder/'samples.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS samples(sid TEXT PRIMARY KEY,data BLOB NOT NULL,checksum TEXT NOT NULL,media TEXT NOT NULL)')
        existing={r[0] for r in db.execute('SELECT sid FROM samples')};planned={r['sample_id'] for r in source['rows']}
        require(existing<=planned,'Unexpected sample in temporal cache')
        bar=Progress(f'EXTRACT {mod} {split} new',len(planned-existing));new=0;started_time=time.perf_counter();counts=[]
        try:
            for row in source['rows']:
                media=check_media(source,row);saved=db.execute('SELECT data,checksum,media FROM samples WHERE sid=?',(row['sample_id'],)).fetchone()
                if saved:
                    require(json.loads(saved[2])==media,'Media changed since sequence cache commit')
                    tokens=payload_decode(saved[0],saved[1])
                else:
                    tokens=encoders.video(row['path']) if mod=='video' else encoders.audio(source,row)
                    validate_tokens(tokens,mod);require(check_media(source,row)==media,'Media changed during extraction')
                    buff=io.BytesIO();np.savez_compressed(buff,**tokens);blob=buff.getvalue()
                    db.execute('INSERT INTO samples VALUES (?,?,?,?)',(row['sample_id'],blob,hashlib.sha256(blob).hexdigest(),json.dumps(media)));db.commit()
                    new+=1;bar.update(new,f'cached_before={len(existing)} completed={len(existing)+new}/{len(planned)}')
                    if new==1 or new%25==0:
                        atomic_json(folder/'progress.json',dict(new=new,resumed=len(existing),completed=len(existing)+new,total=len(planned),seconds=time.perf_counter()-started_time))
                validate_tokens(tokens,mod);counts.append(len(tokens['valid']))
            # Build memory-mapped arrays only after all sample transactions have committed.
            offsets=np.concatenate(([0],np.cumsum(counts))).astype(np.int64);total=int(offsets[-1])
            shapes={'embedding':((total,DIMS[mod]),np.float32),'probabilities':((total,len(CLASSES[mod])),np.float32),
                    'timestamp':((total,),np.float64),'valid':((total,),np.bool_)}
            arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=typ,shape=shape) for k,(shape,typ) in shapes.items()}
            valid_counts=[];pack=Progress(f'PACK {mod} {split}',len(planned))
            for i,row in enumerate(source['rows']):
                blob,checksum=db.execute('SELECT data,checksum FROM samples WHERE sid=?',(row['sample_id'],)).fetchone();t=payload_decode(blob,checksum)
                a,b=map(int,offsets[i:i+2])
                for k in arrays:arrays[k][a:b]=t[k]
                valid_counts.append(int(t['valid'].sum()));pack.update(i+1)
            for arr in arrays.values():arr.flush()
            arrays.clear()
            np.savez(folder/'index.npz',**source['data'],offsets=offsets,valid_counts=np.asarray(valid_counts,np.int64))
            files={name:sha(folder/name) for name in ['index.npz']+[k+'.npy' for k in shapes]}
            stats=dict(n=len(planned),tokens=total,valid_tokens=int(sum(valid_counts)),missing=int(sum(n==0 for n in valid_counts)),
                one_valid_token=int(sum(n==1 for n in valid_counts)),valid_length_min=int(min(valid_counts)),
                valid_length_median=float(np.median(valid_counts)),valid_length_max=int(max(valid_counts)),new=new,resumed=len(existing),
                elapsed_seconds=time.perf_counter()-started_time)
            atomic_json(folder/'COMPLETED.json',dict(identity=identity,files=files,**stats))
            status('EXTRACT',f'{mod} {split} COMPLETE: {stats}')
        finally:db.close()
    return load_packed(folder,identity)
