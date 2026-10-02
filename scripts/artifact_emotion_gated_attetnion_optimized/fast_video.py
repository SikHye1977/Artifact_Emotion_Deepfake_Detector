"""Two video decode passes shared by X3D and batched embedding-only HSEmotion."""
import time
import av
import cv2
import numpy as np
import torch
from PIL import Image
from src.datasets.media import uniform_indices


def shared_video(path, cfg, detector=None):
    if cfg.crop_size < 1 or cfg.resize_short < cfg.crop_size:raise ValueError('Invalid spatial configuration')
    times=[]; timing={}; start=time.perf_counter()
    with av.open(str(path)) as c:
        if not c.streams.video:raise ValueError('Missing video stream')
        stream=c.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=1
        for frame in c.decode(stream):
            if frame.pts is None or frame.time_base is None:raise ValueError('Missing timestamp')
            times.append(float(frame.pts*frame.time_base))
            if len(times)>cfg.max_frames or times[-1]-times[0]>cfg.max_duration:raise ValueError('Media resource guard exceeded')
    timing['timestamp_decode']=time.perf_counter()-start
    indices=uniform_indices(times,cfg.frames);wanted=set(indices.tolist());images={}
    frame_ids=[];token_times=[];valid=[];crops=[];positions=[];detect_seconds=0
    start=time.perf_counter()
    with av.open(str(path)) as c:
        stream=c.streams.video[0];stream.codec_context.thread_count=1
        for i,frame in enumerate(c.decode(stream)):
            if i not in wanted:continue
            if frame.pts is None or float(frame.pts*frame.time_base)!=times[i]:raise ValueError('Timeline changed between decode passes')
            rgb=frame.to_ndarray(format='rgb24');h,w=rgb.shape[:2]
            scale=cfg.resize_short/min(h,w);nw,nh=max(cfg.crop_size,round(w*scale)),max(cfg.crop_size,round(h*scale))
            resized=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)
            x,y=(nw-cfg.crop_size)//2,(nh-cfg.crop_size)//2
            images[i]=resized[y:y+cfg.crop_size,x:x+cfg.crop_size].copy()
            if detector is not None:
                t=time.perf_counter()
                faces=detector.detectMultiScale(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),scaleFactor=1.1,minNeighbors=5,minSize=(40,40))
                detect_seconds+=time.perf_counter()-t
                good=len(faces)==1
                if good:
                    x,y,w,h=map(int,faces[0]);positions.append(len(valid));crops.append(rgb[y:y+h,x:x+w].copy())
                frame_ids.append(i);token_times.append(times[i]);valid.append(good)
    if set(images)!=wanted:raise ValueError('Incomplete decode')
    timing['selected_decode_resize']=time.perf_counter()-start-detect_seconds;timing['face_detection']=detect_seconds
    t=time.perf_counter();array=np.stack([images[int(i)] for i in indices]).astype(np.float32)/255
    video=torch.from_numpy(array).permute(3,0,1,2).contiguous()
    mean=torch.tensor(cfg.mean,dtype=torch.float32)[:,None,None,None];std=torch.tensor(cfg.std,dtype=torch.float32)[:,None,None,None]
    if (std<=0).any():raise ValueError('Invalid normalization')
    video=(video-mean)/std;timing['video_tensor']=time.perf_counter()-t
    return dict(video=video,frame_indices=indices,crops=crops,positions=positions,
                selected_timestamps=np.asarray(times)[np.unique(indices)],token_times=np.asarray(token_times),token_valid=np.asarray(valid,bool),token_frame_ids=np.asarray(frame_ids,np.int64),timing=timing)


@torch.inference_mode()
def batched_tokens(data, extractor, dim, batch_size=32):
    if batch_size<1:raise ValueError('face batch size must be positive')
    features=np.zeros((len(data['token_valid']),dim),np.float32)
    recognizer=extractor.recognizer
    for start in range(0,len(data['crops']),batch_size):
        faces=data['crops'][start:start+batch_size]
        tensor=torch.stack([recognizer.test_transforms(Image.fromarray(face)) for face in faces]).to(recognizer.device)
        values=recognizer.model(tensor).detach().cpu().numpy().reshape(len(faces),dim)
        if not np.isfinite(values).all():raise ValueError('Nonfinite HSE embeddings')
        features[data['positions'][start:start+batch_size]]=values
    return dict(embedding=features,timestamp=data['token_times'],valid=data['token_valid'],frame_index=data['token_frame_ids'])
