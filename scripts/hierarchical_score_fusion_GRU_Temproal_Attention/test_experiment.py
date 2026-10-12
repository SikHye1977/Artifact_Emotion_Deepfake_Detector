"""Tests target time masking, source selection, immutable artifacts and recovery."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .common import DIMS,CLASSES,atomic_json,digest,score_or
from .model import EmotionFlowHead
from .data import subset_indices
from .extract import prepare
from .train import fit

def synthetic(prefix,n=8):
    rng=np.random.default_rng(21);t=3
    return dict(sample_id=np.array([f'fav:{prefix}{i}' for i in range(n)]),group=np.array([f'{prefix}{i}' for i in range(n)]),
        labels=np.array([[i%2]*3 for i in range(n)]),artifact_probability=np.linspace(.1,.9,n),
        embedding=rng.normal(size=(n*t,1280)).astype(np.float32),probabilities=np.full((n*t,8),1/8,np.float32),
        timestamp=np.tile(np.arange(t,dtype=float),n),valid=np.ones(n*t,bool),offsets=np.arange(n+1)*t,valid_counts=np.full(n,t))

class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_padding_is_ignored_and_attention_is_normalized(self):
        torch.manual_seed(3);head=EmotionFlowHead('video').eval()
        e=torch.randn(2,4,1280);p=torch.full((2,4,8),1/8);m=torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool)
        y,a=head(e,p,m);e[1,2:]=float('nan');p[1,2:]=float('nan');other,b=head(e,p,m)
        torch.testing.assert_close(y,other,rtol=0,atol=0);torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(a.sum(1),torch.ones(2));self.assertEqual(float(a[1,2:].sum()),0.)
    def test_all_missing_rejected(self):
        with self.assertRaises(ValueError):EmotionFlowHead('audio')(torch.zeros(1,1,768),torch.ones(1,1,9)/9,torch.zeros(1,1,dtype=torch.bool))
    def test_sequence_order_matters(self):
        torch.manual_seed(9);h=EmotionFlowHead('video').eval();e=torch.randn(2,5,1280);p=torch.ones(2,5,8)/8;m=torch.ones(2,5,dtype=torch.bool)
        a,_=h(e,p,m);b,_=h(e.flip(1),p,m);self.assertGreater(float((a-b).abs().max()),1e-6)
    def test_gradients_reach_gru_attention_and_projection(self):
        h=EmotionFlowHead('audio');y,a=h(torch.randn(3,4,768),torch.ones(3,4,9)/9,torch.ones(3,4,dtype=torch.bool));y.sum().backward()
        for p in h.parameters():self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
    def test_hierarchical_or_algebra(self):
        x=np.random.default_rng(4).uniform(size=(4,50));a=score_or(x[0],x[1]);out=score_or(a,score_or(x[2],x[3]))
        np.testing.assert_allclose(out,1-np.prod(1-x,axis=0));self.assertTrue((out>=a).all())
    def test_smoke_is_deterministic_and_stratified(self):
        d=synthetic('a');d['labels']=np.array([[max(i%2,(i//2)%2),i%2,(i//2)%2] for i in range(8)])
        a=subset_indices(d,4,42);np.testing.assert_array_equal(a,subset_indices(d,4,42));self.assertEqual(len(set(map(tuple,d['labels'][a,1:]))),4)
    def test_exact_training_epoch_resume(self):
        cfg=dict(head_device='cpu',seed=42,epochs=2,batch_size=4,lr=.0005,weight_decay=.0001,pos_weight=.25)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);train,dev=synthetic('train'),synthetic('dev')
            fit(train,dev,'video',d/'full',cfg,'test')
            with self.assertRaises(InterruptedError):fit(train,dev,'video',d/'resume',cfg,'test',interrupt_after_epoch=1)
            fit(train,dev,'video',d/'resume',cfg,'test')
            a=torch.load(d/'full/last.pt',weights_only=True);b=torch.load(d/'resume/last.pt',weights_only=True)
            for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
            for k in a['best_model']:torch.testing.assert_close(a['best_model'][k],b['best_model'][k],rtol=0,atol=0)
            self.assertEqual([x['loss'] for x in a['history']],[x['loss'] for x in b['history']])
    def test_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'overlap'):fit(synthetic('same'),synthetic('same'),'video',d,{},'test')
    def test_extraction_resume_keeps_committed_samples(self):
        class Encoder:
            def __init__(self,fail=False):self.calls=0;self.fail=fail
            def video(self,path):
                self.calls+=1
                if self.fail and self.calls==2:raise InterruptedError('simulate interruption')
                return dict(embedding=np.zeros((128,1280),np.float32),probabilities=np.ones((128,8),np.float32)/8,
                    timestamp=np.arange(128,dtype=float)/25,valid=np.ones(128,bool),frame_index=np.arange(128))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'source';(base/'samples').mkdir(parents=True);data=synthetic('tiny',2);rows=[]
            for sid in data['sample_id']:
                path=root/(str(sid).replace(':','_')+'.bin');path.write_bytes(b'fake fixture media')
                stat=path.stat();atomic_json(base/'samples'/(digest(str(sid))+'.json'),dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns))
                rows.append(dict(sample_id=str(sid),path=str(path)))
            source=dict(folder=base,rows=rows,data={k:data[k] for k in ('sample_id','group','labels','artifact_probability')})
            with self.assertRaises(InterruptedError):prepare(source,'video','train',root/'out',{'test':1},Encoder(True))
            enc=Encoder();actual=prepare(source,'video','train',root/'out',{'test':1},enc,resume=True)
            self.assertEqual(enc.calls,1);np.testing.assert_array_equal(actual['artifact_probability'],data['artifact_probability'])
            with self.assertRaises(ValueError):prepare(source,'video','train',root/'out',{'test':2},enc,resume=True)
            # Release Windows mmap handles before TemporaryDirectory removes this fixture.
            actual.clear()

if __name__=='__main__':unittest.main()
