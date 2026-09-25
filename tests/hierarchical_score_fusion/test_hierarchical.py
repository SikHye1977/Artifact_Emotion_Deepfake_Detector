import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.emotion_deepfake_head import EmotionDeepfakeHead, masked_mean
from src.fusion.hierarchical_score_fusion import probabilistic_or, HierarchicalScoreFusion
from src.branches.hierarchical_artifact_branch import HierarchicalArtifactBranch
from src.branches.hierarchical_emotion_branch import HierarchicalEmotionBranch
from src.datasets.hierarchical_emotion_inputs import audio_window_starts
from src.datasets.hierarchical_emotion_inputs import audio_features, video_features
spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/hierarchical_score_fusion_script.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeModel(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.bias=nn.Parameter(torch.tensor(p))
    def forward(self,x):
        return types.SimpleNamespace(fake_probability=self.bias.expand(len(x)))


class Tests(unittest.TestCase):
    def _shard(self, path, ids):
        path.mkdir(parents=True,exist_ok=True)
        n=len(ids); y=np.arange(n)%2
        np.savez_compressed(path/'cache.npz',sample_id=np.array(ids),group=np.array(ids),
            labels=np.stack([y,y,y],axis=1),video_mean=np.zeros((n,1280),np.float32),
            audio_mean=np.zeros((n,256),np.float32),video_count=np.ones(n,dtype='int64'),
            audio_count=np.ones(n,dtype='int64'),artifact_video=np.full(n,.2),artifact_audio=np.full(n,.3))
        recipe={'synthetic_multi':True}
        runner.write_json(path/'manifest.json',dict(format='hierarchical_cache_v1',split='train',dataset='FAV',n=n,
            recipe=recipe,recipe_sha256=runner.digest(recipe),cache_sha256=runner.sha256(path/'cache.npz')))
        runner.write_json(path/'COMPLETED.json',{'status':'complete'})

    def test_partition_and_merge_order_and_missing(self):
        ids=[f's{i}' for i in range(11)]
        parts=runner.partition_ids(ids,4)
        self.assertEqual([len(p) for p in parts],[3,3,3,2])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); paths=[root/str(i) for i in range(4)]
            for path,subset in zip(paths,parts): self._shard(path,subset)
            output=root/'merged';output.mkdir()
            runner.merge_shards(paths,ids,output,'train')
            data,_=runner.load_cache(output,'train')
            self.assertEqual(data['sample_id'].tolist(),ids)
            with self.assertRaises(ValueError):runner.merge_shards(paths[:-1],ids,output,'train')
            with self.assertRaises(ValueError):runner.merge_shards(paths+[paths[0]],ids,output,'train')

    def test_multi_worker_assignment_failure_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'cfg.json';config.write_text('{}')
            args=types.SimpleNamespace(output=root,config=config,split='train',gpus=['0','1','2','3'],
                  torch_threads=2,allow_test=False)
            ids=[f's{i}' for i in range(12)]
            plan=dict(ids=ids,shards=runner.partition_ids(ids,4))
            calls=[]; failing={'2'}
            owner=self
            class Process:
                pid=123
                def __init__(self,command,env,stdout,stderr):
                    gpu=env['CUDA_VISIBLE_DEVICES'];calls.append(gpu)
                    owner.assertEqual(command[command.index('--device')+1],'cuda:0')
                    self.status=1 if gpu in failing else 0
                    path=Path(command[command.index('--output')+1])
                    selected=Path(command[command.index('--sample-ids')+1]).read_text().splitlines()
                    if self.status==0:owner._shard(path,selected)
                    else:path.mkdir()
                def poll(self):return self.status
            with patch.object(runner,'multi_plan',return_value=plan),patch('subprocess.Popen',Process):
                with self.assertRaises(RuntimeError):runner.cache_multi(args,runner.Logger(root))
                self.assertEqual(calls,['0','1','2','3'])
                failing.clear();calls.clear()
                runner.cache_multi(args,runner.Logger(root))
                self.assertEqual(calls,['2'])
            data,_=runner.load_cache(root,'train')
            self.assertEqual(data['sample_id'].tolist(),ids)

    def test_mask_nan_and_empty(self):
        x=torch.tensor([[[2.,4.],[float('nan'),float('nan')],[4.,6.]]])
        m=torch.tensor([[True,False,True]])
        torch.testing.assert_close(masked_mean(x,m),torch.tensor([[3.,5.]]))
        with self.assertRaises(ValueError): masked_mean(x,torch.zeros_like(m))

    def test_mean_permutation_invariance(self):
        x=torch.randn(3,5,7); m=torch.ones(3,5,dtype=torch.bool)
        torch.testing.assert_close(masked_mean(x,m),masked_mean(x[:,[4,2,0,1,3]],m))

    def test_frozen_artifact_even_train(self):
        b=HierarchicalArtifactBranch(FakeModel(.2),FakeModel(.3)).train()
        self.assertFalse(b.training)
        self.assertTrue(all(not p.requires_grad for p in b.parameters()))
        out=b(torch.zeros(2,3,128,1,1),torch.zeros(2,100))
        torch.testing.assert_close(out['score_artifact'],torch.full((2,),.44))
        with self.assertRaises(ValueError): b(torch.zeros(1,3,16,1,1),torch.zeros(1,100))

    def test_or_equivalence_and_monotonicity(self):
        a,b,c,d=[torch.rand(50) for _ in range(4)]
        artifact=probabilistic_or(a,b)
        total=HierarchicalScoreFusion()(artifact,probabilistic_or(c,d))
        torch.testing.assert_close(total,1-(1-a)*(1-b)*(1-c)*(1-d))
        self.assertTrue((total>=artifact-1e-7).all())
        with self.assertRaises(ValueError): probabilistic_or(torch.tensor([2.]),torch.tensor([.5]))

    def test_head_training_and_roundtrip(self):
        torch.manual_seed(2)
        x=torch.cat([torch.full((8,3),-1.),torch.full((8,3),1.)])
        y=torch.cat([torch.zeros(8),torch.ones(8)])
        head=EmotionDeepfakeHead(3,8,0.)
        with self.assertRaises(RuntimeError): head(x)
        head.fit_standardizer(x)
        mean=head.mean.clone()
        opt=torch.optim.Adam(head.parameters(),lr=.03)
        criterion=nn.BCEWithLogitsLoss()
        first=criterion(head(x),y).item()
        for _ in range(30):
            opt.zero_grad(); loss=criterion(head(x),y); loss.backward(); opt.step()
        self.assertLess(loss.item(),first*.3)
        head.eval()
        clone=EmotionDeepfakeHead(3,8,0.)
        clone.load_state_dict(head.state_dict()); clone.eval()
        torch.testing.assert_close(clone(x),head(x),rtol=0,atol=0)
        torch.testing.assert_close(head.mean,mean)

    def test_windows(self):
        self.assertEqual(audio_window_starts(1000),[0])
        self.assertEqual(audio_window_starts(48240),[0])
        self.assertEqual(audio_window_starts(64240),[0,16000])
        self.assertEqual(audio_window_starts(70000),[0,16000,21760])

    def test_audio_short_padding_and_silence(self):
        observed=[]
        def mel(chunk, **kwargs):
            observed.append(chunk.copy())
            return np.zeros((300,40),np.float32)
        ps=types.SimpleNamespace(logfbank=mel,delta=lambda a,n: a)
        b=types.SimpleNamespace(audio_head=types.SimpleNamespace(input_dim=256),
                                extract_audio=lambda x: np.ones((len(x),256),np.float32))
        with patch.dict(sys.modules,{'python_speech_features':ps}):
            mean,n,detail=audio_features(np.ones(1000,np.float32),b)
            self.assertEqual(n,1)
            self.assertEqual(detail[0]['padding_samples'],47240)
            self.assertEqual(len(observed[0]),48240)
            self.assertTrue((observed[0][1000:]==0).all())
            _,n,_=audio_features(np.zeros(1000,np.float32),b)
            self.assertEqual(n,0)

    def test_video_unique_indices_and_invalid_face(self):
        class Frame:
            def __init__(self,i): self.pts=i; self.time_base=.04; self.i=i
            def to_ndarray(self,format): return np.full((8,8,3),self.i,np.uint8)
        class Container:
            streams=types.SimpleNamespace(video=[types.SimpleNamespace(codec_context=types.SimpleNamespace(thread_count=0))])
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def decode(self,s): return iter([Frame(0),Frame(1),Frame(2)])
        detector=types.SimpleNamespace(detectMultiScale=lambda gray,**k: [(0,0,8,8)] if gray[0,0,0]!=1 else [])
        b=types.SimpleNamespace(video_head=types.SimpleNamespace(input_dim=1280),
           extract_video=lambda rgb: np.full(1280,rgb[0,0,0],np.float32))
        with patch.dict(sys.modules,{'av':types.SimpleNamespace(open=lambda p:Container()),
                  'cv2':types.SimpleNamespace(COLOR_RGB2GRAY=0,cvtColor=lambda rgb,k:rgb)}):
            mean,n,details=video_features('dummy',np.linspace(0,2,128).astype('int64'),b,detector)
        self.assertEqual(n,2)
        self.assertEqual(len(details),3)
        np.testing.assert_array_equal(mean,np.ones(1280))

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            runner.no_overlap({'sample_id':['a'],'group':['same']},{'sample_id':['b'],'group':['same']})

    def test_metrics(self):
        out=runner.metrics(np.array([0,0,1,1]),np.array([.1,.6,.4,.9]))
        self.assertEqual((out['tn'],out['fp'],out['fn'],out['tp']),(1,1,1,1))
        self.assertEqual(out['accuracy'],.5)

    def test_cli_train_evaluate_and_test_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            for split in ['train','dev','test']:
                path=tmp/split; path.mkdir()
                y=np.tile([0,1],8)
                rng=np.random.default_rng(42)
                values=dict(sample_id=np.array([f'{split}-{i}' for i in range(16)]),
                  group=np.array([f'{split}-g{i}' for i in range(16)]), labels=np.stack([y,y,y],1),
                  video_mean=(rng.normal(0,.1,(16,1280))+(2*y[:,None]-1)).astype('float32'),
                  audio_mean=(rng.normal(0,.1,(16,256))+(2*y[:,None]-1)).astype('float32'),
                  video_count=np.ones(16,dtype='int64'),audio_count=np.ones(16,dtype='int64'),
                  artifact_video=np.where(y,.7,.2),artifact_audio=np.where(y,.7,.2))
                # Keep one absent face to verify coverage, without feeding zero as real.
                values['video_count'][0]=0; values['video_mean'][0]=0
                np.savez_compressed(path/'cache.npz',**values)
                runner.write_json(path/'manifest.json',dict(format='hierarchical_cache_v1',split=split,dataset='FAV',n=16,
                    recipe={'synthetic':True},recipe_sha256=runner.digest({'synthetic':True}),
                    cache_sha256=runner.sha256(path/'cache.npz')))
            script=str(ROOT/'scripts/hierarchical_score_fusion_script.py')
            result=subprocess.run([sys.executable,script,'train','--train-cache',str(tmp/'train'),
               '--dev-cache',str(tmp/'dev'),'--epochs','2','--output',str(tmp/'run')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((tmp/'run/COMPLETED.json').exists())
            report=json.loads((tmp/'run/dev_metrics.json').read_text())
            self.assertEqual(report['coverage']['common_valid'],15)
            command=[sys.executable,script,'evaluate','--cache',str(tmp/'test'),'--checkpoint',str(tmp/'run/emotion_heads.pt'),
                     '--split','test','--output',str(tmp/'blocked')]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            command[-1]=str(tmp/'evaluation'); command.append('--allow-test')
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((tmp/'evaluation/COMPLETED.json').exists())
            # Cache mutation must be detected.
            with (tmp/'test/cache.npz').open('ab') as f: f.write(b'altered')
            with self.assertRaises(ValueError): runner.load_cache(tmp/'test','test')


if __name__=='__main__': unittest.main()
