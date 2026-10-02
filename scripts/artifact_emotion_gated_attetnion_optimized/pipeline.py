"""Resumable stage orchestration; interrupted short head training starts a new attempt."""
from pathlib import Path
import argparse,subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.artifact_emotion_gated_attetnion_optimized.storage import WriterLock,atomic_json
from scripts.artifact_emotion_gated_attetnion_optimized.common import read,load_cache,sha,implementation
from scripts.artifact_emotion_gated_attetnion_optimized.experiment import load_head


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
 p.add_argument('--encoder-device',default='cuda:0');p.add_argument('--head-device',default='cpu')
 p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--face-batch-size',type=int,default=1);p.add_argument('--legacy-video-train',type=Path)
 args=p.parse_args();root=args.output.resolve()
 if root.exists() and not args.resume:raise FileExistsError('Use a new run or --resume')
 root.mkdir(parents=True,exist_ok=True)
 config=dict(encoder_device=args.encoder_device,head_device=args.head_device,epochs=args.epochs,seed=args.seed,
             face_batch_size=args.face_batch_size,legacy_video_train=str(args.legacy_video_train.resolve()) if args.legacy_video_train else None,implementation=implementation())
 script=Path(__file__).with_name('experiment.py')
 def run(*parts):
  command=[sys.executable,str(script),*[str(x) for x in parts]]
  print('STAGE:',subprocess.list2cmdline(command),flush=True);subprocess.run(command,check=True,cwd=ROOT)
 with WriterLock(root):
  if (root/'RUN.json').exists():
   if read(root/'RUN.json')!=config:raise ValueError('Run settings/source changed; use a new output')
  else:atomic_json(root/'RUN.json',config)
  run('preflight','--trusted-artifact-checkpoints')
  def extraction(mod,split,lock=None):
   path=root/'cache'/mod/split
   if (path/'COMPLETED.json').exists():
    _,meta=load_cache(path,mod,split)
    if meta['recipe']['implementation']!=implementation():raise ValueError('Cached extraction source changed')
    return
   parts=['extract','--modality',mod,'--split',split,'--device',args.encoder_device,'--trusted-artifact-checkpoints','--output',path,'--face-batch-size',args.face_batch_size]
   if path.exists():parts+=['--resume']
   if lock:parts+=['--lock',lock]
   if args.legacy_video_train and mod=='video' and split=='train':parts+=['--reuse-legacy-tokens',args.legacy_video_train]
   run(*parts)
  heads={}
  for mod in ('video','audio'):
   for split in ('train','dev'):extraction(mod,split)
   attempt=1
   while True:
    path=root/'heads'/f'{mod}_{attempt:03d}'
    if not path.exists():
     run('train','--modality',mod,'--train-cache',root/'cache'/mod/'train','--dev-cache',root/'cache'/mod/'dev','--output',path,'--device',args.head_device,'--epochs',args.epochs,'--seed',args.seed)
    if (path/'COMPLETED.json').exists():
     _,state=load_head(path/'best.pt')
     if sha(path/'best.pt')!=read(path/'COMPLETED.json')['checkpoint_sha256']:raise ValueError('Head changed')
     for split,key in [('train','train_cache_sha256'),('dev','dev_cache_sha256')]:
      if state[key]!=sha(root/'cache'/mod/split/'features.npz'):raise ValueError('Head/cache mismatch')
     heads[mod]=path/'best.pt';break
    print(f'Incomplete head attempt preserved: {path}; training will restart in a new attempt',flush=True);attempt+=1
  lock=root/'LOCKED_BEFORE_TEST.json'
  if not lock.exists():run('lock','--video-head',heads['video'],'--audio-head',heads['audio'],'--output',lock)
  frozen=read(lock)
  if frozen['implementation']!=implementation():raise ValueError('Lock source changed')
  for mod in heads:
   if frozen['branches'][mod]['sha256']!=sha(heads[mod]):raise ValueError('Lock/head mismatch')
   extraction(mod,'test',lock)
  evaluation=root/'test';attempt=1
  while evaluation.exists() and not (evaluation/'COMPLETED.json').exists():
   print(f'Incomplete evaluation preserved: {evaluation}',flush=True)
   attempt+=1;evaluation=root/f'test_{attempt:03d}'
  if not (evaluation/'COMPLETED.json').exists():
   if evaluation.exists():raise ValueError('Interrupted evaluation output exists; preserve it and use experiment.py evaluate with a fresh output')
   run('evaluate','--video-cache',root/'cache/video/test','--audio-cache',root/'cache/audio/test','--lock',lock,'--split','test','--device',args.head_device,'--output',evaluation)
  elif read(evaluation/'COMPLETED.json')['lock_sha256']!=sha(lock):raise ValueError('Evaluation lock changed')
  print('PIPELINE COMPLETE',root,flush=True)

if __name__=='__main__':main()
