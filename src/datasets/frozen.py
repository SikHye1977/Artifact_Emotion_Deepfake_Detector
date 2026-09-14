"""Map-style PyTorch Dataset using frozen manifests and sample-id quality records."""
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .media import MediaConfig,read_video,read_audio

ROOT=Path(__file__).resolve().parents[2]

class FrozenMediaDataset(Dataset):
    def __init__(self,dataset,split,config=None,sample_ids=None,eligibility_path=None,task='paired',registry=None):
        if split not in ('train','dev','test') or task not in ('video','audio','paired'):raise ValueError('Invalid split/task')
        if eligibility_path is None:raise ValueError('Explicit sample-id eligibility required; uninspected files are not approved')
        self.config=config or MediaConfig();self.rows=[];self.task=task
        reg=Path(registry) if registry else ROOT/'configs/fixed_manifests.json'
        entry=json.loads(reg.read_text())['datasets'][dataset];manifest=ROOT/entry['manifest']
        digest=hashlib.sha256()
        with manifest.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=entry['sha256']:raise ValueError('Frozen manifest hash mismatch')
        evidence={}
        with Path(eligibility_path).open() as f:
            for line in f:
                r=json.loads(line)
                if r['manifest_sha256']!=entry['sha256'] and r['dataset']==dataset:raise ValueError('Eligibility manifest mismatch')
                if r['dataset']==dataset:evidence[r['sample_id']]=r
        selected=set(sample_ids) if sample_ids is not None else None
        with manifest.open() as f:
            for line in f:
                r=json.loads(line)
                if r['split']!=split or (selected is not None and r['sample_id'] not in selected):continue
                e=evidence.get(r['sample_id'])
                if e is None or not e['eligible'][task]:continue
                self.rows.append(dict(sample_id=r['sample_id'],path=r['video_path'],split=split,dataset=dataset,
                    labels=[r['clip_label'],r['video_label'],r['audio_label']],size=e['size'],mtime_ns=e['mtime_ns']))
        if selected is not None and {r['sample_id'] for r in self.rows}!=selected:
            raise ValueError('Requested samples not all eligible in this fixed split')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];st=Path(row['path']).stat()
        if st.st_size!=row['size'] or st.st_mtime_ns!=row['mtime_ns']:raise ValueError('Media changed after quality audit')
        output=dict(row)
        if self.task!='audio':output.update(read_video(row['path'],self.config))
        else:
            n=self.config.frames;h=self.config.crop_size
            output.update(video=torch.zeros(3,n,h,h),video_mask=torch.zeros(n,dtype=torch.bool),frame_unique_mask=torch.zeros(n,dtype=torch.bool),frame_indices=torch.full((n,),-1,dtype=torch.int64),timestamps=torch.full((n,),float('nan'),dtype=torch.float64))
        if self.task!='video':output['audio'],output['has_audio']=read_audio(row['path'],self.config)
        else:output['audio'],output['has_audio']=torch.empty(0,dtype=torch.float32),False
        output['labels']=torch.tensor([-1 if x is None else x for x in row['labels']],dtype=torch.int64)
        output['label_mask']=torch.tensor([x is not None for x in row['labels']],dtype=torch.bool)
        return output
