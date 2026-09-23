"""128 uniform unique video frames; complete RAVDESS CRNN windows with a fixed 1 s hop."""
import numpy as np
import torch


def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    area=max(0,x2-x1)*max(0,y2-y1)
    return area/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area,1)


def visual(path, extractor, detector):
    import av
    import cv2
    timestamps=[]
    with av.open(str(path)) as c:
        s=c.streams.video[0];s.codec_context.thread_count=1
        for frame in c.decode(s):
            if frame.pts is None: raise ValueError('No frame PTS')
            timestamps.append(float(frame.pts*frame.time_base))
            if len(timestamps)>100000 or timestamps[-1]-timestamps[0]>300:
                raise ValueError('Media resource guard exceeded')
    if not timestamps or (np.diff(timestamps)<0).any(): raise ValueError('Invalid video timeline')
    selected=np.unique(np.linspace(0,len(timestamps)-1,128).astype('int64'))
    wanted=set(selected.tolist());seen=set();probs=[];times=[];valid=[];segments=[];boxes=[];indices=[]
    previous=None;segment=-1
    with av.open(str(path)) as c:
        s=c.streams.video[0];s.codec_context.thread_count=1
        for i,frame in enumerate(c.decode(s)):
            if i not in wanted: continue
            seen.add(i); t=float(frame.pts*frame.time_base)
            if t!=timestamps[i]:raise ValueError('Video changed between decode passes')
            if times and t<=times[-1]:continue
            image=frame.to_ndarray(format='rgb24')
            faces=detector.detectMultiScale(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),scaleFactor=1.1,minNeighbors=5,minSize=(40,40))
            good=len(faces)==1; p=np.full(8,np.nan,np.float32);box=[-1]*4
            if good:
                x,y,w,h=map(int,faces[0]);box=[x,y,x+w,y+h]
                if previous is None or iou(previous,box)<.2:segment+=1
                previous=box
                p=extractor.extract(image[y:y+h,x:x+w].copy())['probabilities']
            else:previous=None
            probs.append(p);times.append(t);valid.append(good);segments.append(segment if good else -1);boxes.append(box);indices.append(i)
    if seen!=wanted:raise ValueError('Incomplete video decode')
    return dict(values=np.asarray(probs,np.float32),times=np.asarray(times)-times[0],valid=np.asarray(valid),
                segments=np.asarray(segments),boxes=np.asarray(boxes),frame_indices=np.asarray(indices))


def auditory(waveform,extractor,device):
    """Complete 3-second windows at 1-second hops; no artificial repeated windows."""
    signal=waveform.detach().cpu()
    if signal.ndim!=1 or not len(signal) or not torch.isfinite(signal).all():
        raise ValueError('Expected finite mono waveform')
    values=[];times=[];valid=[];starts=[]
    for start in range(0,max(0,len(signal)-48000+1),16000):
        chunk=signal[start:start+48000];good=bool(torch.any(chunk!=0))
        p=np.full(8,np.nan,np.float32)
        if good:
            with torch.inference_mode():
                p=extractor(chunk[None,None].to(device))['probabilities'][0].detach().cpu().numpy().copy()
        values.append(p);valid.append(good);times.append((start+24000)/16000);starts.append(start)
    return dict(values=np.asarray(values,np.float32).reshape(-1,8),times=np.asarray(times),
                valid=np.asarray(valid,dtype=bool),starts=np.asarray(starts,dtype='int64'))


def auditory_seed(waveform,extractor,window_samples=48000,hop_samples=16000):
    """Nine-class probabilities over complete windows; no padding/repetition."""
    signal=waveform.detach().cpu().numpy() if hasattr(waveform,'detach') else np.asarray(waveform)
    if signal.ndim!=1 or not len(signal) or not np.isfinite(signal).all():raise ValueError('Invalid audio')
    if window_samples<=0 or hop_samples<=0:raise ValueError('Invalid window/hop')
    values=[];times=[];valid=[];starts=[]
    for start in range(0,max(0,len(signal)-window_samples+1),hop_samples):
        chunk=signal[start:start+window_samples]
        good=bool(np.any(chunk!=0))
        values.append(extractor.probabilities(chunk) if good else np.full(9,np.nan,np.float32))
        valid.append(good);starts.append(start);times.append((start+window_samples/2)/16000)
    return dict(values=np.asarray(values,np.float32).reshape(-1,9),times=np.asarray(times,dtype=float),
                valid=np.asarray(valid,dtype=bool),starts=np.asarray(starts,dtype='int64'))
