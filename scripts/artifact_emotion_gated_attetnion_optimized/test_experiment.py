"""Synthetic functional tests; no raw datasets or frozen encoders needed."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from .model import GatedBranch,score_or
from .common import write,sha,digest,implementation,load_cache,disjoint
from .features import pool_tokens,audio_chunks,auditory_tokens
from .experiment import main


def make_cache(root, modality, split, reverse=False):
    root.mkdir(parents=True)
    n=12;rng=np.random.default_rng(42)
    video=np.arange(n)%2;audio=(np.arange(n)//2)%2
    labels=np.stack([video|audio,video,audio],axis=1).astype('int64')
    ids=np.array([f'{split}:{i}' for i in range(n)])
    data=dict(sample_id=ids,group=np.array([f'{split}:group:{i}' for i in range(n)]),labels=labels,
        artifact=rng.normal(size=(n,7)).astype('float32'),emotion=rng.normal(size=(n,9)).astype('float32'),
        emotion_valid=np.arange(n)%3!=0,artifact_probability=np.linspace(.1,.9,n))
    if reverse:data={k:v[::-1].copy() for k,v in data.items()}
    np.savez_compressed(root/'features.npz',**data)
    recipe=dict(dataset='FAV',modality=modality,registry_sha256='registry',eligibility_sha256='eligibility',
                manifest_sha256='manifest',implementation=implementation())
    write(root/'manifest.json',dict(format='artifact_emotion_embedding_v1',modality=modality,split=split,
          n=n,artifact_dim=7,emotion_dim=9,recipe=recipe,recipe_sha256=digest(recipe),sha256=sha(root/'features.npz')))
    write(root/'COMPLETED.json',dict(n=n))
    return data


def test_gate_shapes_fallback_and_gradients():
    torch.manual_seed(1);m=GatedBranch(7,9,dropout=0)
    a=torch.randn(3,7,requires_grad=True);e=torch.randn(3,9,requires_grad=True)
    with torch.no_grad():e[0]=float('nan')
    logits,g=m(a,e,torch.tensor([False,True,True]))
    assert logits.shape==(3,) and g.shape==(3,128)
    assert torch.equal(g[0],torch.ones(128)) and torch.isfinite(logits).all()
    expected=m.classifier(m.artifact_projection(a[:1])).squeeze(-1)
    torch.testing.assert_close(logits[:1],expected)
    logits.sum().backward()
    assert torch.equal(e.grad[0],torch.zeros(9))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.gate.weight.grad.abs().sum()>0


def test_embedding_pool_and_short_audio():
    got=pool_tokens([[1,3],[float('nan'),float('nan')],[3,5]],[True,False,True],2)
    np.testing.assert_array_equal(got,[2,4])
    np.testing.assert_array_equal(pool_tokens(np.zeros((0,2)),[],2),[0,0])
    chunks=list(audio_chunks(np.ones(32000)))
    assert len(chunks)==1 and len(chunks[0][1])==32000
    assert len(list(audio_chunks(np.ones(64000))))==2
    class FakeNative:
        def generate(self,**kwargs):
            assert kwargs['extract_embedding'] is True and kwargs['granularity']=='utterance'
            return [{'feats':np.arange(768,dtype=np.float32),'scores':np.zeros(9)}]
    class Wrapper:model=FakeNative()
    result=auditory_tokens(np.ones(32000,dtype=np.float32),Wrapper())
    assert result['embedding'].shape==(1,768) and result['end_sample'][0]==32000


def test_cache_tamper_and_group_leak(tmp_path):
    make_cache(tmp_path/'train','video','train');d,_=load_cache(tmp_path/'train','video','train')
    with pytest.raises(ValueError,match='overlap'):disjoint(d,d)
    with (tmp_path/'train/features.npz').open('ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'):load_cache(tmp_path/'train','video','train')


def test_end_to_end_independent_heads_and_or(tmp_path, capsys):
    for mod in ('video','audio'):
        for split in ('train','dev','test'):make_cache(tmp_path/mod/split,mod,split,reverse=mod=='audio')
        assert main(['train','--modality',mod,'--train-cache',str(tmp_path/mod/'train'),'--dev-cache',str(tmp_path/mod/'dev'),
                     '--output',str(tmp_path/mod/'head'),'--epochs','2','--batch-size','4'])==0
        state=torch.load(tmp_path/mod/'head/best.pt',weights_only=True)
        assert state['training']['label_index']==(1 if mod=='video' else 2)
    lock=tmp_path/'lock.json'
    main(['lock','--video-head',str(tmp_path/'video/head/best.pt'),'--audio-head',str(tmp_path/'audio/head/best.pt'),'--output',str(lock)])
    main(['evaluate','--video-cache',str(tmp_path/'video/test'),'--audio-cache',str(tmp_path/'audio/test'),
          '--split','test','--lock',str(lock),'--output',str(tmp_path/'evaluation')])
    import csv
    rows=list(csv.DictReader((tmp_path/'evaluation/predictions.csv').open()))
    assert len(rows)==12 and all(r['included']=='True' for r in rows)
    for row in rows:
        expected=1-(1-float(row['video_score']))*(1-float(row['audio_score']))
        assert abs(float(row['fused_score'])-expected)<1e-7
    report=json.loads((tmp_path/'evaluation/metrics.json').read_text())
    assert report['coverage']['video_emotion_missing']==4
    console = capsys.readouterr().out
    assert 'TRAIN video epoch 1/2' in console and 'TRAIN audio epoch 2/2' in console
    assert 'batches (100.0%)' in console and 'loss=' in console and 'ETA=' in console
    assert 'BEST checkpoint saved' in console and 'EVALUATE test video' in console
    # Other-modality and clip labels must not affect a video head's training.
    altered=tmp_path/'video/train/features.npz'
    with np.load(altered) as z:changed={k:z[k] for k in z.files}
    changed['labels'][:,0]=1-changed['labels'][:,0];changed['labels'][:,2]=1-changed['labels'][:,2]
    np.savez_compressed(altered,**changed)
    meta=json.loads((altered.parent/'manifest.json').read_text());meta['sha256']=sha(altered);write(altered.parent/'manifest.json',meta)
    main(['train','--modality','video','--train-cache',str(altered.parent),'--dev-cache',str(tmp_path/'video/dev'),
          '--output',str(tmp_path/'video/head_other_labels'),'--epochs','2','--batch-size','4'])
    old=torch.load(tmp_path/'video/head/best.pt',weights_only=True)['model']
    new=torch.load(tmp_path/'video/head_other_labels/best.pt',weights_only=True)['model']
    for key in old:torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
    # Lock remains immutable and detects checkpoint changes.
    with (tmp_path/'video/head/best.pt').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='Locked head changed'):
        main(['evaluate','--video-cache',str(tmp_path/'video/test'),'--audio-cache',str(tmp_path/'audio/test'),
              '--split','test','--lock',str(lock),'--output',str(tmp_path/'bad')])


def test_or_is_probability_rule():
    np.testing.assert_allclose(score_or(np.array([0,.4,1]),np.array([1,.4,0])),[1,.64,1])


def test_preflight_distinguishes_blocked_and_pass(monkeypatch, capsys):
    import importlib
    features = importlib.import_module('scripts.artifact_emotion_gated_attetnion_optimized.features')
    monkeypatch.setattr(features, 'dependencies', lambda modality, trusted: dict(modality=modality, issues=['Missing media audit']))
    assert main(['preflight']) == 2
    output = capsys.readouterr().out
    assert 'video: BLOCKED' in output and 'audio: BLOCKED' in output
    assert 'training has NOT started' in output
    monkeypatch.setattr(features, 'dependencies', lambda modality, trusted: dict(modality=modality, issues=[]))
    assert main(['preflight']) == 0
    assert 'PASS (exit code 0)' in capsys.readouterr().out


def test_new_local_inputs_need_no_source_hashes(tmp_path, monkeypatch):
    from . import local_inputs
    monkeypatch.setattr(local_inputs, 'ROOT', tmp_path)
    records=[];quality=[]
    for part in ('train','dev','test'):
        media=tmp_path/(part+'.mp4');media.write_bytes(b'local test fixture')
        records.append(dict(sample_id=part,source_id=part,split=part,video_path=str(media),clip_label=1,video_label=0,audio_label=1))
        quality.append(dict(dataset='FAV',sample_id=part,eligible={'video':True,'audio':True},manifest_sha256='different historical hash',size=-1,mtime_ns=-1))
    (tmp_path/'manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    (tmp_path/'eligibility.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in quality))
    write(tmp_path/'registry.json',{'datasets':{'FAV':{'manifest':'manifest.jsonl','sha256':'different historical hash'}}})
    rows=local_inputs.local_rows(tmp_path/'registry.json',tmp_path/'eligibility.jsonl','video')
    assert len(rows)==3
    for row in rows:local_inputs.verify_local_file(row)
    records[1]['source_id']=records[0]['source_id']
    (tmp_path/'manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    with pytest.raises(ValueError,match='Group leakage'):
        local_inputs.local_rows(tmp_path/'registry.json',tmp_path/'eligibility.jsonl','video')
    (tmp_path/'train.mp4').unlink()
    with pytest.raises(FileNotFoundError):local_inputs.verify_local_file(rows[0])
