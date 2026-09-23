"""Raw-media emotion preprocessing; never reuses normalized X3D tensors."""
import numpy as np


def audio_window_starts(length, window=48240, hop=16000):
    if min(length, window, hop) <= 0:
        raise ValueError('Positive lengths required')
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, hop))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def audio_features(waveform, branch, device='cpu'):
    import python_speech_features as ps
    import torch
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not len(signal) or not np.isfinite(signal).all():
        raise ValueError('Empty/nonfinite audio')
    features, details = [], []
    for start in audio_window_starts(len(signal)):
        chunk = signal[start:start + 48240]
        valid = bool(np.any(chunk != 0))
        details.append({'start_sample': start, 'real_samples': len(chunk),
                        'padding_samples': 48240 - len(chunk), 'valid': valid})
        if not valid:
            continue
        chunk = np.pad(chunk, (0, 48240 - len(chunk)))
        mel = ps.logfbank(chunk, samplerate=16000, winlen=.025, winstep=.01,
                         nfilt=40, nfft=512, lowfreq=0, highfreq=8000, preemph=.97)
        delta = ps.delta(mel, 2)
        x = np.stack([mel, delta, ps.delta(delta, 2)]).astype(np.float32)
        if x.shape != (3, 300, 40):
            raise ValueError(f'ACRNN acoustic input mismatch: {x.shape}')
        features.append(branch.extract_audio(torch.from_numpy(x[None]).to(device))[0])
    mean = np.mean(features, axis=0) if features else np.zeros(branch.audio_head.input_dim, np.float32)
    return mean, len(features), details


def video_features(path, frame_indices, branch, detector):
    import av
    import cv2
    ids = np.asarray(frame_indices, dtype=np.int64)
    if ids.shape != (128,) or (ids < 0).any():
        raise ValueError('128 real frame indices from the X3D decoder required')
    wanted = set(ids.tolist())
    features, details, seen = [], [], set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index not in wanted:
                continue
            seen.add(index)
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Missing video timestamp')
            rgb = frame.to_ndarray(format='rgb24')
            found = detector.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                                             scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            item = {'frame_index': index, 'timestamp': float(frame.pts * frame.time_base),
                    'valid': len(found) == 1, 'face_count': len(found)}
            if len(found) == 1:
                x, y, w, h = map(int, found[0])
                item['box_xyxy'] = [x, y, x+w, y+h]
                features.append(branch.extract_video(rgb[y:y+h, x:x+w].copy()))
            details.append(item)
            if seen == wanted:
                break
    if seen != wanted:
        raise ValueError('Emotion decoder did not reproduce X3D frame indices')
    mean = np.mean(features, axis=0) if features else np.zeros(branch.video_head.input_dim, np.float32)
    return mean, len(features), details
