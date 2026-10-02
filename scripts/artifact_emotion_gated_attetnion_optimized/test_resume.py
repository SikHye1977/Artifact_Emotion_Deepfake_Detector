from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from .storage import SampleStore,WriterLock
from .common import digest,read,load_cache


def test_atomic_sample_resume_corruption_and_input_change(tmp_path):
    media=tmp_path/'media';media.write_bytes(b'media')
    row=dict(sample_id='sample',group='g',labels=[1,1,0],path=str(media))
    root=tmp_path/'cache';root.mkdir()
    store=SampleStore(root,{'recipe':1},'train',[row],False)
    tokens=dict(embedding=np.ones((2,4),np.float32),timestamp=np.array([0.,1.]),valid=np.array([True,True]))
    store.save(row,np.ones(3),.5,tokens,np.ones(4))
    second=SampleStore(root,{'recipe':1},'train',[row],True)
    assert second.load(row)['artifact'].shape==(3,)
    with pytest.raises(ValueError,match='recipe'):
        SampleStore(root,{'recipe':2},'train',[row],True)
    sample=root/'samples'/(digest('sample')+'.npz')
    sample.write_bytes(b'crashed write')
    assert second.load(row) is None
    store.save(row,np.ones(3),.5,tokens,np.ones(4))
    media.write_bytes(b'changed media')
    with pytest.raises(ValueError,match='media changed'):second.load(row)


def test_writer_lock_releases_after_exception(tmp_path):
    with pytest.raises(RuntimeError,match='abort'):
        with WriterLock(tmp_path):
            with pytest.raises((RuntimeError,OSError)):
                with WriterLock(tmp_path):pass
            raise RuntimeError('abort')
    with WriterLock(tmp_path):pass


def test_full_extract_stops_and_resumes_without_recomputing(tmp_path,monkeypatch):
    from . import features,local_inputs,fast_video
    import src.models.hsemotion as hse_module
    registry=tmp_path/'registry.json';manifest=tmp_path/'manifest.jsonl';eligibility=tmp_path/'eligibility.jsonl'
    manifest.write_text('manifest');eligibility.write_text('quality')
    import json
    registry.write_text(json.dumps({'datasets':{'FAV':{'manifest':str(manifest)}}}))
    checkpoint=tmp_path/'weight.pt';checkpoint.write_bytes(b'weight')
    inputs=[]
    for i in range(3):
        path=tmp_path/f'{i}.mp4';path.write_bytes(b'media')
        inputs.append(dict(sample_id=str(i),group=str(i),labels=[i%2,i%2,0],path=str(path)))
    monkeypatch.setattr(features,'dependencies',lambda *args:dict(issues=[],checkpoint=str(checkpoint),registry=str(registry),eligibility=str(eligibility)))
    monkeypatch.setattr(local_inputs,'local_rows',lambda *args:inputs)
    class Artifact(torch.nn.Module):
        def forward(self,x):return SimpleNamespace(features=torch.ones(1,3),fake_probability=torch.tensor([.5]))
    cfg={'video':{'num_frames':128,'crop_size':32,'resize_short_side':32,'normalization':{'mean':[0,0,0],'std':[1,1,1]}}}
    monkeypatch.setattr(local_inputs,'load_local_artifact',lambda *args:(Artifact(),cfg))
    class Emotion:
        def __init__(self,device):
            self.checkpoint_path=checkpoint
            self.recognizer=SimpleNamespace(classifier_weights=np.zeros((8,4)),model=torch.nn.Linear(1,1))
    monkeypatch.setattr(hse_module,'EmotionExtractor',Emotion)
    calls=[]
    def decode(path,*args):calls.append(str(path));return {'video':torch.ones(3,2,4,4)}
    monkeypatch.setattr(fast_video,'shared_video',decode)
    monkeypatch.setattr(fast_video,'batched_tokens',lambda *args:dict(embedding=np.ones((2,4),np.float32),timestamp=np.array([0.,1.]),valid=np.array([True,True]),frame_index=np.array([0,1])))
    args=SimpleNamespace(modality='video',split='train',trusted_artifact_checkpoints=True,device='cpu',lock=None,
        output=tmp_path/'output',resume=False,face_batch_size=32,reuse_legacy_tokens=None,stop_after=1)
    features.extract(args)
    assert len(calls)==1 and not (args.output/'COMPLETED.json').exists()
    args.resume=True;args.stop_after=0;features.extract(args)
    assert len(calls)==3
    data,meta=load_cache(args.output,'video','train')
    assert meta['n']==3 and list(data['sample_id'])==['0','1','2']
    features.extract(args)
    assert len(calls)==3
