"""Frozen raw-media encoders. No emotion probabilities/change statistics in head inputs."""
from pathlib import Path
import importlib.metadata
import json
import numpy as np
import torch
from .common import ROOT, DEFAULT_CHECKPOINTS, sha, read, write, digest, implementation
from src.utils.local_paths import load_local_paths, project_path
from .progress import Progress, status


def dependencies(modality, trusted):
    if not trusted:
        raise ValueError('Use --trusted-artifact-checkpoints for your own migrated training checkpoints')
    model_name = 'x3d' if modality == 'video' else 'aasist'
    checkpoint = DEFAULT_CHECKPOINTS / model_name / 'best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('format') not in ('x3a_ddp_v1', 'x3a_single_v1'):
        raise ValueError('Unexpected artifact checkpoint format')
    prov = state['provenance']; cfg = prov['config']; issues = []
    registry = ROOT / cfg['data']['split_manifest']
    quality = read(ROOT / cfg['data']['eligibility'])
    eligibility = ROOT / quality['eligibility_manifest']
    from .local_inputs import local_rows, verify_local_file
    counts = {}
    try:
        rows = local_rows(registry, eligibility, modality)
        for row in rows:
            verify_local_file(row)
            counts[row['split']] = counts.get(row['split'], 0) + 1
        for split in ('train','dev','test'):
            if not counts.get(split):issues.append('No eligible local rows for ' + split)
    except Exception as error:
        issues.append('Local data: ' + str(error))
    local = load_local_paths()
    if modality == 'video':
        from src.models.hsemotion import hsemotion_checkpoint, HSE_SHA256
        weight = hsemotion_checkpoint()
        if not weight.is_file() or sha(weight) != HSE_SHA256:
            issues.append('Missing/changed HSEmotion official weight')
    else:
        from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import download_seed
        try:
            if not local.get('emotion2vec_snapshot'):
                raise ValueError('Configure verified local emotion2vec_snapshot; no implicit download')
            download_seed()
        except Exception as error:
            issues.append('Seed installation: ' + str(error))
    return dict(modality=modality, checkpoint=str(checkpoint), registry=str(registry),
                eligibility=str(eligibility), input_policy='new_local_experiment', samples_by_split=counts, issues=issues)


def audio_chunks(signal, window=48000, hop=16000):
    """Complete 3s windows; short clips use their full real waveform, never padding."""
    n = len(signal)
    starts = [0] if n < window else list(range(0, n - window + 1, hop))
    for start in starts:
        yield start, signal[start:min(start + window, n)]


def pool_tokens(tokens, valid, dim):
    tokens = np.asarray(tokens, dtype=np.float32).reshape(-1, dim)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (len(tokens),) or not np.isfinite(tokens[valid]).all():
        raise ValueError('Invalid token embeddings')
    return tokens[valid].mean(axis=0) if valid.any() else np.zeros(dim, dtype=np.float32)


def visual_tokens(path, indices, extractor, detector, dim):
    import av, cv2
    wanted = set(np.asarray(indices).tolist()); seen = set()
    tokens, times, valid, frame_ids = [], [], [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]; stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index not in wanted:
                continue
            if frame.pts is None:
                raise ValueError('Missing frame timestamp')
            seen.add(index); rgb = frame.to_ndarray(format='rgb24')
            faces = detector.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            good = len(faces) == 1; feature = np.zeros(dim, np.float32)
            if good:
                x, y, w, h = map(int, faces[0])
                feature = extractor.extract(rgb[y:y+h, x:x+w].copy())['features']
            tokens.append(feature); times.append(float(frame.pts * frame.time_base)); valid.append(good); frame_ids.append(index)
    if seen != wanted:
        raise ValueError('Face decoder did not reproduce artifact frame indices')
    return dict(embedding=np.asarray(tokens, np.float32), timestamp=np.asarray(times),
                valid=np.asarray(valid, bool), frame_index=np.asarray(frame_ids, np.int64))


def auditory_tokens(signal, model, dim=768):
    tokens, times, valid, starts, ends = [], [], [], [], []
    for start, chunk in audio_chunks(signal):
        good = len(chunk) >= 400 and bool(np.any(chunk != 0))
        feature = np.zeros(dim, np.float32)
        if good:
            result = model.model.generate(input=chunk, granularity='utterance', extract_embedding=True)
            if len(result) != 1 or 'feats' not in result[0]:
                raise ValueError('Seed did not return an embedding')
            feature = np.asarray(result[0]['feats'], np.float32).reshape(-1)
            if feature.shape != (dim,) or not np.isfinite(feature).all():
                raise ValueError('Invalid seed embedding')
        tokens.append(feature); valid.append(good); starts.append(start); ends.append(start+len(chunk))
        times.append((start+len(chunk)/2)/16000)
    return dict(embedding=np.asarray(tokens, np.float32).reshape(-1,dim), timestamp=np.asarray(times),
                valid=np.asarray(valid, bool), start_sample=np.asarray(starts), end_sample=np.asarray(ends))


def _extract_locked(args):
    from .local_inputs import local_rows, verify_local_file, load_local_artifact
    from src.datasets.media import MediaConfig, read_video, read_audio
    stage = f'EXTRACT {args.modality} {args.split}'
    status(stage, 'Checking extraction prerequisites...')
    dep = dependencies(args.modality, args.trusted_artifact_checkpoints)
    if dep['issues']:
        status(stage, 'BLOCKED: no media extraction started')
        raise ValueError('Extraction blocked:\n' + '\n'.join(dep['issues']))
    if args.split == 'test' and args.lock is None:
        raise ValueError('Test extraction requires --lock from locked video/audio heads')
    lock = read(args.lock) if args.lock else None
    if lock and (lock['format'] != 'gated_or_lock_v1' or lock['implementation'] != implementation()):
        raise ValueError('Invalid/stale lock implementation')
    status(stage, f'Loading frozen encoders and eligibility records on {args.device}...')
    torch.set_num_threads(4)
    model, cfg = load_local_artifact(Path(dep['checkpoint']), args.modality)
    registry, eligibility = Path(dep['registry']), Path(dep['eligibility'])
    model.to(args.device).eval().requires_grad_(False)
    rows_input = local_rows(registry, eligibility, args.modality, args.split)
    entry = read(registry)['datasets']['FAV']
    groups = {row['sample_id']:row['group'] for row in rows_input}
    recipe = dict(dataset='FAV', modality=args.modality, artifact_checkpoint_sha256=sha(dep['checkpoint']),
        registry_sha256=sha(registry), eligibility_sha256=sha(eligibility), manifest_sha256=sha(ROOT/entry['manifest']),
        input_policy='new_local_experiment', extraction_version='resumable_batched_v1', face_batch_size=args.face_batch_size, implementation=implementation(), pooling='mean_valid_emotion_embeddings',
        packages={name:importlib.metadata.version(name) for name in ['torch','numpy','av','opencv-python-headless','hsemotion','funasr','timm']})
    if args.modality == 'video':
        import cv2
        from src.models.hsemotion import EmotionExtractor
        from src.utils.local_paths import file_sha256
        emotion = EmotionExtractor(device=args.device); emotion.recognizer.model.requires_grad_(False)
        dim = int(emotion.recognizer.classifier_weights.shape[1])
        detector_path = Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(str(detector_path))
        if detector.empty():raise ValueError('Missing Haar face detector')
        cv2.setNumThreads(1); v = cfg['video']
        media = MediaConfig(frames=v['num_frames'], crop_size=v['crop_size'], resize_short=v['resize_short_side'],
            mean=tuple(v['normalization']['mean']), std=tuple(v['normalization']['std']))
        recipe.update(artifact_input=v, emotion_weight_sha256=file_sha256(emotion.checkpoint_path),
                      face_detector_sha256=sha(detector_path), emotion_input='HSE native RGB face preprocessing; unique artifact frame indices')
    else:
        from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import FrozenEmotion2VecSeed, download_seed
        snapshot, revision = download_seed(); emotion = FrozenEmotion2VecSeed(snapshot,args.device); dim = 768
        if cfg['audio'].get('sample_rate') != 16000 or cfg['audio']['target_num_samples'] != 284672:
            raise ValueError('Unexpected AASIST training waveform recipe')
        media = MediaConfig(sample_rate=16000, audio_target_num_samples=cfg['audio']['target_num_samples'])
        recipe.update(artifact_input=cfg['audio'], emotion_pretrained=emotion.provenance, revision=revision,
                      emotion_input='16k mono; 3s complete windows hop1s; short clip uses whole unpadded waveform; native layer_norm; utterance embedding')
    recipe_hash = digest(recipe)
    if lock and recipe_hash != lock['branches'][args.modality]['recipe_sha256']:
        raise ValueError('Test extraction recipe differs from locked training recipe')
    from .storage import SampleStore, atomic_json, atomic_npz
    from .fast_video import shared_video, batched_tokens
    from .common import load_cache
    if (args.output/'COMPLETED.json').exists():
        load_cache(args.output,args.modality,args.split)
        if read(args.output/'manifest.json')['recipe_sha256']!=recipe_hash:raise ValueError('Completed recipe changed')
        status(stage,'Already complete; cache verified')
        return
    store=SampleStore(args.output,recipe,args.split,rows_input,args.resume)
    (args.output/'tokens').mkdir(exist_ok=True)
    legacy=args.reuse_legacy_tokens
    if legacy:
        if args.modality!='video':raise ValueError('Legacy token reuse only supports video')
        old=read(legacy/'STARTED.json')
        if old['split']!=args.split:raise ValueError('Legacy split differs')
        for key in ('dataset','modality','artifact_checkpoint_sha256','manifest_sha256','artifact_input','emotion_weight_sha256','face_detector_sha256','emotion_input'):
            if old['recipe'][key]!=recipe[key]:raise ValueError('Legacy token recipe differs: '+key)
        atomic_json(args.output/'legacy_token_source.json',dict(path=str(legacy.resolve()),started_sha256=sha(legacy/'STARTED.json')))
    rows=[];token_hashes={};reused=0;imported=0;computed=0
    status(stage,f'Encoders ready; resumable output={args.output}')
    progress=Progress(stage,len(rows_input))
    with torch.inference_mode():
        for n,row in enumerate(rows_input,1):
            verify_local_file(row);before=Path(row['path']).stat()
            saved=store.load(row)
            if saved is not None:
                reused+=1
                tokens={k[6:]:v for k,v in saved.items() if k.startswith('token_')}
            else:
                if args.modality=='video':
                    tokens=read_legacy_tokens(legacy,row['sample_id'],dim) if legacy else None
                    data=shared_video(row['path'],media,None if tokens is not None else detector)
                    if tokens is not None:
                        if not np.array_equal(tokens['frame_index'],np.unique(data['frame_indices'])) or not np.array_equal(tokens['timestamp'],data['selected_timestamps']):
                            raise ValueError('Legacy tokens do not match decoded timeline')
                        imported+=1
                    else:tokens=batched_tokens(data,emotion,dim,args.face_batch_size)
                    prediction=model(data['video'].unsqueeze(0).to(args.device))
                else:
                    wave,present=read_audio(row['path'],media)
                    if not present:raise ValueError('Missing eligible audio')
                    padded=torch.nn.functional.pad(wave,(0,media.audio_target_num_samples-len(wave)))
                    prediction=model(padded.unsqueeze(0).to(args.device))
                    tokens=auditory_tokens(wave.numpy(),emotion,dim)
                artifact=prediction.features[0].cpu().numpy().astype(np.float32)
                pooled=pool_tokens(tokens['embedding'],tokens['valid'],dim)
                after=Path(row['path']).stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Media changed during extraction')
                saved=store.save(row,artifact,float(prediction.fake_probability[0].cpu()),tokens,pooled);computed+=1
            sid=row['sample_id'];filename=digest(sid)+'.npz'
            token_path=args.output/'tokens'/filename
            # Reconstruct exports from the committed sample, including after an interrupted export.
            atomic_npz(token_path,sample_id=np.asarray(sid),**tokens);token_hashes[filename]=sha(token_path)
            rows.append(dict(sample_id=sid,group=groups[sid],labels=[-1 if v is None else v for v in row['labels']],
                artifact=saved['artifact'],emotion=saved['emotion'],emotion_valid=bool(saved['emotion_valid']),
                artifact_probability=float(saved['artifact_probability'])))
            if n==1 or n%25==0 or n==len(rows_input):
                atomic_json(args.output/'progress.json',dict(done=n,total=len(rows_input),resumed=reused,legacy_tokens=imported,new_samples=computed))
            progress.update(n,f'resumed={reused} legacy_tokens={imported} new={computed}')
            if args.stop_after and computed>=args.stop_after and n<len(rows_input):
                status(stage,'Intentional stop after committed samples; rerun with --resume')
                return
    status(stage,'Writing complete pooled cache...')
    payload={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
    atomic_npz(args.output/'features.npz',**payload)
    atomic_json(args.output/'manifest.json',dict(format='artifact_emotion_embedding_v1',modality=args.modality,split=args.split,
        n=len(rows),artifact_dim=payload['artifact'].shape[1],emotion_dim=dim,recipe=recipe,recipe_sha256=recipe_hash,
        sha256=sha(args.output/'features.npz'),token_sha256=token_hashes))
    atomic_json(args.output/'COMPLETED.json',dict(n=len(rows)))
    status(stage,f'COMPLETE samples={len(rows)} resumed={reused} legacy_tokens={imported}')


def read_legacy_tokens(root,sid,dim):
    import zipfile
    path=root/'tokens'/(digest(sid)+'.npz')
    if not path.exists():return None
    try:
        before=path.stat()
        with np.load(path,allow_pickle=False) as z:
            if str(z['sample_id'])!=sid:raise ValueError('Legacy sample ID mismatch')
            data={k:z[k].copy() for k in ('embedding','timestamp','valid','frame_index')}
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):return None
    except (OSError,EOFError,zipfile.BadZipFile):return None
    n=len(data['valid'])
    if data['embedding'].shape!=(n,dim) or data['timestamp'].shape!=(n,) or data['frame_index'].shape!=(n,) or data['valid'].dtype!=np.bool_:
        raise ValueError('Invalid legacy token shape')
    if not np.isfinite(data['embedding']).all() or not np.isfinite(data['timestamp']).all():raise ValueError('Nonfinite legacy tokens')
    return data


def extract(args):
    from .storage import WriterLock
    if args.face_batch_size<1 or args.stop_after<0:raise ValueError('Invalid batch/stop size')
    if args.reuse_legacy_tokens and args.face_batch_size!=1:raise ValueError('Legacy token reuse requires --face-batch-size 1 for matching numerical behavior')
    if args.output.exists() and not args.resume:raise FileExistsError('Use a new output or --resume')
    args.output.mkdir(parents=True,exist_ok=True)
    with WriterLock(args.output):
        _extract_locked(args)
