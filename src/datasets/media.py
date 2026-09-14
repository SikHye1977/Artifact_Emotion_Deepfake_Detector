"""Full-clip, timestamp-aware media decoding. No model execution."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import av
import cv2
import numpy as np
import torch
from .transforms import pad_audio

@dataclass(frozen=True)
class MediaConfig:
    frames: int = 128
    crop_size: int = 224
    resize_short: int = 256
    sample_rate: int = 16000
    max_duration: float = 300.0
    max_frames: int = 100000
    mean: tuple = (0.0,0.0,0.0)
    std: tuple = (1.0,1.0,1.0)
    audio_target_num_samples: int | None = None


def uniform_indices(times,count):
    if count<1 or len(times)==0:raise ValueError('Positive frame count and nonempty timestamps required')
    ts=np.asarray(times,dtype=np.float64)
    if not np.isfinite(ts).all() or (np.diff(ts)<0).any():raise ValueError('Invalid/nonmonotonic video timestamps')
    indices=np.linspace(0,len(ts)-1,count).astype(np.int64)
    assert len(indices)==count and indices.min()>=0 and indices.max()<len(ts)
    return indices


def read_video(path,cfg):
    if cfg.crop_size<1 or cfg.resize_short<cfg.crop_size:raise ValueError('Invalid spatial configuration')
    times=[]
    with av.open(str(path)) as container:
        if not container.streams.video:raise ValueError('No video stream')
        stream=container.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=1
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:raise ValueError('Missing video timestamps')
            times.append(float(frame.pts*frame.time_base))
            if len(times)>cfg.max_frames or times[-1]-times[0]>cfg.max_duration:raise ValueError('Media resource guard exceeded')
    indices=uniform_indices(times,cfg.frames);wanted=set(indices.tolist());images={}
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        for i,frame in enumerate(container.decode(stream)):
            if i in wanted:
                rgb=frame.to_ndarray(format='rgb24');h,w=rgb.shape[:2]
                scale=cfg.resize_short/min(h,w);nw,nh=max(cfg.crop_size,round(w*scale)),max(cfg.crop_size,round(h*scale))
                resized=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)
                x,y=(nw-cfg.crop_size)//2,(nh-cfg.crop_size)//2
                images[i]=resized[y:y+cfg.crop_size,x:x+cfg.crop_size]
    if len(images)!=len(wanted):raise ValueError('Frame count changed between decode passes')
    array=np.stack([images[int(i)] for i in indices]).astype(np.float32)/255.0
    video=torch.from_numpy(array).permute(3,0,1,2).contiguous()
    mean=torch.tensor(cfg.mean,dtype=torch.float32)[:,None,None,None]
    std=torch.tensor(cfg.std,dtype=torch.float32)[:,None,None,None]
    if (std<=0).any():raise ValueError('Normalization std must be positive')
    first=np.concatenate(([True],indices[1:]!=indices[:-1]))
    return dict(video=(video-mean)/std,video_mask=torch.ones(cfg.frames,dtype=torch.bool),
                frame_unique_mask=torch.from_numpy(first),frame_indices=torch.from_numpy(indices.copy()),
                timestamps=torch.tensor(np.asarray(times)[indices],dtype=torch.float64),decoded_frames=len(times))


def read_audio(path,cfg):
    with av.open(str(path)) as container:
        if not container.streams.audio:return torch.empty(0,dtype=torch.float32),False
        if container.duration is not None and container.duration/av.time_base>cfg.max_duration:raise ValueError('Audio duration guard exceeded')
    p=subprocess.run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(path),'-map','0:a:0','-vn',
                      '-ac','1','-ar',str(cfg.sample_rate),'-f','f32le','-'],capture_output=True,timeout=120,check=True)
    data=np.frombuffer(p.stdout,dtype='<f4').copy()
    if len(data)==0 or not np.isfinite(data).all():raise ValueError('Empty or nonfinite audio')
    if len(data)>cfg.max_duration*cfg.sample_rate:raise ValueError('Audio samples exceed guard')
    waveform = torch.from_numpy(data)
    if cfg.audio_target_num_samples is not None:
        if len(waveform)>cfg.audio_target_num_samples:
            raise ValueError('Truncation is prohibited')
    return waveform,True


def collate_media(items, audio_target_num_samples=None):
    if not items:raise ValueError('Empty batch')
    result={key:[r[key] for r in items] for key in ('sample_id','path','split','dataset')}
    result['decoded_frames']=[r.get('decoded_frames',0) for r in items]
    for key in ('video','video_mask','frame_unique_mask','frame_indices','timestamps','labels','label_mask'):
        result[key]=torch.stack([r[key] for r in items])
    lengths=torch.tensor([len(r['audio']) for r in items],dtype=torch.int64)
    target = int(lengths.max()) if audio_target_num_samples is None else audio_target_num_samples
    if target < int(lengths.max()):raise ValueError('Truncation is prohibited')
    result['audio_lengths']=lengths;result['audio']=torch.zeros((len(items),target),dtype=torch.float32)
    result['audio_mask']=torch.zeros_like(result['audio'],dtype=torch.bool)
    for i,r in enumerate(items):
        n=len(r['audio']);result['audio'][i,:n]=r['audio'];result['audio_mask'][i,:n]=True
    result['has_audio']=torch.tensor([r['has_audio'] for r in items],dtype=torch.bool)
    return result
