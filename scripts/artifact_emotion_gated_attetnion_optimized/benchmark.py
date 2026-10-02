"""Paired video benchmark while the original experiment remains running."""
from pathlib import Path
import sys,time,json,csv,gc,argparse
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.artifact_emotion_gated_attetnion_optimized.common import DEFAULT_CHECKPOINTS,write,digest
from scripts.artifact_emotion_gated_attetnion_optimized.local_inputs import load_local_artifact
from scripts.artifact_emotion_gated_attetnion_optimized.fast_video import shared_video,batched_tokens
from scripts.artifact_emotion_gated_attetnion_optimized.features import pool_tokens,auditory_tokens
from scripts.artifact_emotion_gated_attetnion_optimized.storage import SampleStore
from scripts.artifact_emotion_gated_attetnion.features import visual_tokens
from src.datasets.media import MediaConfig,read_video,read_audio
from src.models.hsemotion import EmotionExtractor


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=8);p.add_argument('--face-batch-size',type=int,default=1);p.add_argument('--video-only',action='store_true');args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(4);import cv2;cv2.setNumThreads(1)
 torch.backends.cudnn.benchmark=False
 candidates=[r for r in csv.DictReader((ROOT/'reports/dataset_duration_100_seed42_20260928/measurements.csv').open(encoding='utf-8')) if r['dataset']=='FAV' and r['status']=='ok']
 training={}
 with (ROOT/'reports/fixed_splits_v1_release/FAV/split_manifest.jsonl').open(encoding='utf-8') as f:
  for line in f:
   r=json.loads(line)
   if r['split']=='train':training[r['sample_id']]=r
 candidates=sorted([r for r in candidates if 'fav:'+r['relative_path'] in training],key=lambda r:float(r['video_seconds']))
 selected=[candidates[int(i)] for i in np.linspace(0,len(candidates)-1,args.samples)]
 results=dict(method='Duration-stratified subset of previous seed42 random sample; paired alternating order; one warmup; shared GPU/CPU with active production extraction; not isolated throughput',video=[],audio=[],head_timing=[],face_batch_size=args.face_batch_size)
 write(args.output/'selection.json',selected)
 model,cfg=load_local_artifact(DEFAULT_CHECKPOINTS/'x3d/best.pt','video');model.to('cuda:0')
 emotion=EmotionExtractor('cuda:0');emotion.recognizer.model.requires_grad_(False)
 detector=cv2.CascadeClassifier(str(Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'))
 v=cfg['video'];media=MediaConfig(frames=v['num_frames'],crop_size=v['crop_size'],resize_short=v['resize_short_side'],mean=tuple(v['normalization']['mean']),std=tuple(v['normalization']['std']))
 def timed(call):
  torch.cuda.synchronize();t=time.perf_counter();out=call();torch.cuda.synchronize();return out,time.perf_counter()-t
 def baseline(path):
  data,t_decode=timed(lambda:read_video(path,media))
  pred,t_art=timed(lambda:model(data['video'][None].to('cuda:0')))
  tok,t_em=timed(lambda:visual_tokens(path,data['frame_indices'].numpy(),emotion,detector,1280))
  return dict(artifact=pred.features.cpu().numpy(),tokens=tok,video=data['video'],timing={'video_decode':t_decode,'artifact':t_art,'emotion_decode_detect_infer':t_em})
 def fast(path):
  data,t_decode=timed(lambda:shared_video(path,media,detector))
  pred,t_art=timed(lambda:model(data['video'][None].to('cuda:0')))
  tok,t_em=timed(lambda:batched_tokens(data,emotion,1280,args.face_batch_size))
  return dict(artifact=pred.features.cpu().numpy(),tokens=tok,video=data['video'],timing=dict(data['timing'],artifact=t_art,emotion_batch=t_em,shared_decode_total=t_decode))
 with torch.inference_mode():
  warm=fast(selected[0]['local_path']);del warm
  for i,row in enumerate(selected):
   path=row['local_path'];outputs={};times={}
   for label,fn in ([('baseline',baseline),('optimized',fast)] if i%2==0 else [('optimized',fast),('baseline',baseline)]):
    outputs[label],times[label]=timed(lambda fn=fn:fn(path))
   a,b=outputs['baseline'],outputs['optimized']
   exact_video=torch.equal(a['video'],b['video'])
   eq_mask=all(np.array_equal(a['tokens'][k],b['tokens'][k]) for k in ('valid','timestamp','frame_index'))
   diff=float(np.max(np.abs(a['tokens']['embedding']-b['tokens']['embedding'])))
   artifact_diff=float(np.max(np.abs(a['artifact']-b['artifact'])))
   close=np.allclose(a['tokens']['embedding'],b['tokens']['embedding'],rtol=1e-4,atol=5e-5)
   # Saved legacy emotion tokens allow bypassing detection and HSE inference.
   def reuse():
    data=shared_video(path,media,None)
    return model(data['video'][None].to('cuda:0')).features.cpu().numpy()
   _,reuse_time=timed(reuse)
   sample_row=dict(sample_id='fav:'+row['relative_path'],group='benchmark',labels=[1,1,1],path=path)
   scratch=args.output/('storage_'+str(i));scratch.mkdir()
   store=SampleStore(scratch,{'benchmark':1},'benchmark',[sample_row],False)
   _,save_time=timed(lambda:store.save(sample_row,b['artifact'][0],.5,b['tokens'],pool_tokens(b['tokens']['embedding'],b['tokens']['valid'],1280)))
   _,resume_time=timed(lambda:store.load(sample_row))
   item=dict(sample_id=sample_row['sample_id'],duration=float(row['video_seconds']),baseline_seconds=times['baseline'],optimized_seconds=times['optimized'],legacy_reuse_seconds=reuse_time,commit_seconds=save_time,resume_read_seconds=resume_time,
       baseline_stages=a['timing'],optimized_stages=b['timing'],video_tensor_exact=exact_video,token_mask_time_exact=eq_mask,artifact_max_abs=artifact_diff,emotion_max_abs=diff,emotion_allclose=bool(close))
   results['video'].append(item);write(args.output/'results.json',results)
   print(json.dumps(item),flush=True)
   del outputs,a,b
  del model,emotion;gc.collect();torch.cuda.empty_cache()
  if args.video_only:return
  from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import FrozenEmotion2VecSeed,download_seed
  model,cfg=load_local_artifact(DEFAULT_CHECKPOINTS/'aasist/best.pt','audio');model.to('cuda:0')
  snapshot,_=download_seed();emotion=FrozenEmotion2VecSeed(snapshot,'cuda:0')
  ac=MediaConfig(sample_rate=16000,audio_target_num_samples=284672)
  for row in selected:
   def audio_work():
    wave,ok=read_audio(row['local_path'],ac)
    if not ok:raise ValueError('Missing audio')
    pred=model(torch.nn.functional.pad(wave,(0,284672-len(wave)))[None].to('cuda:0'))
    tok=auditory_tokens(wave.numpy(),emotion)
    return len(tok['valid'])
   windows,elapsed=timed(audio_work)
   results['audio'].append(dict(sample_id='fav:'+row['relative_path'],seconds=elapsed,windows=windows))
   write(args.output/'results.json',results);print('AUDIO',elapsed,flush=True)
  del model,emotion;gc.collect();torch.cuda.empty_cache()
 # Synthetic CPU head timing: no scientific training outputs or metrics.
 from scripts.artifact_emotion_gated_attetnion_optimized.model import GatedBranch
 for name,ad,ed in [('video',2048,1280),('audio',160,768)]:
  torch.manual_seed(42);head=GatedBranch(ad,ed);opt=torch.optim.AdamW(head.parameters(),lr=.001)
  a=torch.randn(64,ad);e=torch.randn(64,ed);mask=torch.ones(64,dtype=torch.bool);target=torch.arange(64).float()%2
  elapsed=[]
  for batch in range(60):
   t=time.perf_counter();opt.zero_grad();logits,_=head(a,e,mask);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);opt.step()
   if batch>=10:elapsed.append(time.perf_counter()-t)
  head.eval();t=time.perf_counter()
  with torch.inference_mode():
   for _ in range(13):head(torch.randn(256,ad),torch.randn(256,ed),torch.ones(256,dtype=torch.bool))
  dev_seconds=time.perf_counter()-t
  results['head_timing'].append(dict(modality=name,synthetic_batch64_seconds=float(np.mean(elapsed)),projected_30_epochs_seconds=30*(231*np.mean(elapsed)+dev_seconds),note='CPU synthetic compute estimate; excludes checkpoint/history writes'))
 write(args.output/'results.json',results)
 print('BENCHMARK COMPLETE',flush=True)

if __name__=='__main__':main()
