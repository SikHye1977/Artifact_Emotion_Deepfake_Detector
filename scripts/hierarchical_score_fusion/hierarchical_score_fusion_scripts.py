"""FAV hierarchical OR experiment: cache -> train emotion heads -> evaluate.
Backbones are frozen. This is pipeline orchestration, NOT joint backbone training.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, f1_score, confusion_matrix)
from src.models.hierarchical_score_fusion.hierarchical_checkpoint_loading import sha256
from src.branches.hierarchical_score_fusion.hierarchical_emotion_branch import HierarchicalEmotionBranch
from src.fusion.hierarchical_score_fusion.hierarchical_score_fusion import HierarchicalScoreFusion, probabilistic_or


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Logger:
    def __init__(self, output):
        self.output = output
        self.started = time.monotonic()

    def __call__(self, event, **values):
        row = dict(event=event, elapsed_seconds=round(time.monotonic()-self.started, 2), **values)
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with (self.output / 'progress.jsonl').open('a') as stream:
            stream.write(line + '\n')
        write_json(self.output / 'heartbeat.json', row)


def code_hashes():
    paths = [Path(__file__)]
    for relative in ('src/models/hierarchical_score_fusion/emotion_deepfake_head.py', 'src/models/hierarchical_score_fusion/hierarchical_checkpoint_loading.py',
                     'src/branches/hierarchical_score_fusion/hierarchical_artifact_branch.py', 'src/branches/hierarchical_score_fusion/hierarchical_emotion_branch.py',
                     'src/fusion/hierarchical_score_fusion/hierarchical_score_fusion.py', 'src/datasets/hierarchical_score_fusion/hierarchical_emotion_inputs.py'):
        paths.append(ROOT / relative)
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def cache(args, log):
    import cv2
    from src.datasets.frozen import FrozenMediaDataset
    from src.datasets.media import MediaConfig, read_audio, collate_media
    from src.datasets.hierarchical_score_fusion.hierarchical_emotion_inputs import video_features, audio_features
    from src.models.hierarchical_score_fusion.hierarchical_checkpoint_loading import load_artifact
    from src.models.hsemotion import EmotionExtractor
    from src.models.crnn import AudioEmotionExtractor
    from src.branches.hierarchical_score_fusion.hierarchical_artifact_branch import HierarchicalArtifactBranch

    cfg = json.loads(args.config.read_text())
    if args.split == 'test' and not args.allow_test:
        raise ValueError('Test extraction requires --allow-test after locking the recipe')
    resolve = lambda x: ROOT / x
    # Require existing HSE weights; do not silently download a new model.
    hse_path = Path.home() / '.hsemotion/enet_b0_8_best_afew.pt'
    if not hse_path.is_file():
        raise FileNotFoundError(f'Install the existing verified HSE weight first: {hse_path}')
    x3d, vc, registry, eligibility = load_artifact(resolve(cfg['x3d_checkpoint']), 'video', ROOT, args.trusted_artifact_checkpoints)
    aasist, ac, ar, ae = load_artifact(resolve(cfg['aasist_checkpoint']), 'audio', ROOT, args.trusted_artifact_checkpoints)
    if sha256(registry) != sha256(ar) or sha256(eligibility) != sha256(ae):
        raise ValueError('Artifact checkpoints use different split/eligibility')
    target = ac['audio']['target_num_samples']
    if not isinstance(target, int) or target <= 0:
        raise ValueError('AASIST must record the fixed train-maximum waveform length')
    v = vc['video']
    video_cfg = MediaConfig(frames=128, crop_size=v['crop_size'], resize_short=v['resize_short_side'],
                            mean=tuple(v['normalization']['mean']), std=tuple(v['normalization']['std']))
    if ac['audio'].get('sample_rate', 16000) != 16000:
        raise ValueError('Expected a 16 kHz AASIST training recipe')
    raw_cfg = MediaConfig(sample_rate=16000, audio_target_num_samples=None)
    selected = None
    if args.sample_ids:
        selected = [s.strip() for s in args.sample_ids.read_text().splitlines() if s.strip()]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError('Sample IDs must be unique and nonempty')
    # paired eligibility selects a common media cohort; video-only access avoids decoding audio twice.
    selection = FrozenMediaDataset('FAV', args.split, raw_cfg, sample_ids=selected,
                                   eligibility_path=eligibility, task='paired', registry=registry)
    ids = [r['sample_id'] for r in selection.rows]
    if not ids:
        raise ValueError('Empty eligible cohort')
    ds = FrozenMediaDataset('FAV', args.split, video_cfg, sample_ids=ids,
                           eligibility_path=eligibility, task='video', registry=registry)
    if [r['sample_id'] for r in ds.rows] != ids:
        raise ValueError('Dataset row order mismatch')
    entry = json.loads(registry.read_text())['datasets']['FAV']
    records = {}
    with resolve(entry['manifest']).open() as f:
        for line in f:
            r = json.loads(line)
            if r['sample_id'] in records:
                raise ValueError('Duplicate frozen sample ID')
            records[r['sample_id']] = r
    group_key = cfg.get('group_key', 'source_id')
    for sid in ids:
        if not isinstance(records[sid].get(group_key), (str, int)) or str(records[sid][group_key]) == '':
            raise ValueError(f'Missing verified group key {group_key}: {sid}')
    # Process backbones serially at batch 1; no multi-GPU training or hidden AMP.
    device = args.device
    artifact = HierarchicalArtifactBranch(x3d, aasist).to(device)
    hse = EmotionExtractor(device=device)
    acrnn = AudioEmotionExtractor(resolve(cfg['acrnn_checkpoint']),
             source_file=resolve(cfg.get('acrnn_source', 'third_party/amsdf_acrnn/ACRNN.py')),
             state_dict_key=cfg.get('acrnn_state_dict_key'), device=device)
    emotion = HierarchicalEmotionBranch(hse, acrnn)
    cv2.setNumThreads(1)
    detector_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(str(detector_path))
    if detector.empty():
        raise RuntimeError('Missing Haar face detector')
    recipe = {'registry_sha256': sha256(registry), 'eligibility_sha256': sha256(eligibility),
              'manifest_sha256': entry['sha256'], 'group_key': group_key,
              'x3d_checkpoint_sha256': sha256(resolve(cfg['x3d_checkpoint'])),
              'aasist_checkpoint_sha256': sha256(resolve(cfg['aasist_checkpoint'])),
              'hse_weights_sha256': sha256(hse_path), 'hse_model': hse.model_name,
              'acrnn_weights_sha256': sha256(resolve(cfg['acrnn_checkpoint'])),
              'acrnn_source_sha256': sha256(acrnn.source_file), 'detector_sha256': sha256(detector_path),
              'video_recipe': v, 'aasist_recipe': ac['audio'],
              'emotion_recipe': {'video': 'unique X3D 128 indices; Haar single-face crop; HSE native normalization',
                 'audio': '16k mono; 48240 samples; hop16000; final end-anchored; right-zero-pad short',
                 'acoustic': 'logfbank40/25ms/10ms/512fft/preemph0.97; delta2+delta-delta2; no normalization',
                 'pooling': 'mean of valid embeddings', 'acrnn_pretraining_recipe_verified': False},
              'sources': {str(p.relative_to(ROOT)): sha256(p) for p in
                   [ROOT / ('src/models/' + n + '.py') for n in ('x3d', 'aasist', 'hsemotion', 'crnn')]},
              'implementation': code_hashes(),
              'packages': {name: importlib.metadata.version(name) for name in
                           ('torch', 'numpy', 'av', 'hsemotion', 'python_speech_features', 'timm')},
              'opencv_version': cv2.__version__}
    recipe['sources'].update({str(p.relative_to(ROOT)): sha256(p) for p in
          [ROOT/'src/datasets/media.py', ROOT/'src/datasets/frozen.py']})
    write_json(args.output/'recipe.json', recipe)
    log('cache_start', split=args.split, samples=len(ds), device=device, aasist_max_samples=target)
    rows = []
    for i in range(len(ds)):
        sample = ds[i]  # Existing media stat verification and video recipe.
        batch = collate_media([sample])
        raw_audio, present = read_audio(sample['path'], raw_cfg)
        if not present or raw_audio.ndim != 1 or len(raw_audio) == 0:
            raise ValueError(f'Invalid raw mono audio: {sample["sample_id"]}')
        if len(raw_audio) > target:
            raise ValueError('AASIST train-maximum exceeded; truncation is forbidden')
        padded = torch.nn.functional.pad(raw_audio, (0, target-len(raw_audio)))[None]
        with torch.no_grad():
            scores = artifact(batch['video'].to(device), padded.to(device))
        frame_ids = sample['frame_indices'].cpu().numpy()
        expected_ids = np.linspace(0, sample['decoded_frames']-1, 128).astype(np.int64)
        if not np.array_equal(frame_ids, expected_ids):
            raise ValueError('Existing decoder is not the specified uniform 128-frame recipe')
        vm, vn, vd = video_features(sample['path'], frame_ids, emotion, detector)
        am, an, ad = audio_features(raw_audio.cpu().numpy(), emotion, device)
        labels = sample['labels'].cpu().numpy()
        sid = sample['sample_id']
        rows.append(dict(sample_id=sid, group=str(records[sid][group_key]), labels=labels,
                         video_mean=vm, audio_mean=am, video_count=vn, audio_count=an,
                         artifact_video=float(scores['video_score'].item()),
                         artifact_audio=float(scores['audio_score'].item())))
        with (args.output/'observations.jsonl').open('a') as f:
            f.write(json.dumps(dict(sample_id=sid, video=vd, audio=ad), allow_nan=False)+'\n')
        log('cache_sample', current=i+1, total=len(ds), sample_id=sid,
            video_valid=vn, audio_valid=an, exclusion=None if vn and an else 'missing_emotion_observation')
    fields = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    np.savez_compressed(args.output/'cache.npz', **fields)
    write_json(args.output/'manifest.json', dict(format='hierarchical_cache_v1', dataset='FAV',
        split=args.split, n=len(rows), recipe=recipe, recipe_sha256=digest(recipe),
        cache_sha256=sha256(args.output/'cache.npz'),
        versions={'torch': str(torch.__version__), 'numpy': np.__version__}))
    log('cache_complete', n=len(rows))


def load_cache(path, split):
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['format'] != 'hierarchical_cache_v1' or manifest['split'] != split:
        raise ValueError('Wrong cache format/split')
    if sha256(path/'cache.npz') != manifest['cache_sha256'] or digest(manifest['recipe']) != manifest['recipe_sha256']:
        raise ValueError('Cache hash mismatch')
    with np.load(path/'cache.npz', allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    n = manifest['n']
    if n == 0 or any(len(v) != n for v in result.values()) or len(set(result['sample_id'])) != n:
        raise ValueError('Missing, duplicate or inconsistent sample rows')
    if result['labels'].shape != (n, 3) or not np.isin(result['labels'], [-1, 0, 1]).all():
        raise ValueError('Expected [clip,video,audio] labels with -1 for unknown')
    for name, dim in [('video_mean', 1280), ('audio_mean', 256)]:
        if result[name].shape != (n, dim) or not np.isfinite(result[name]).all():
            raise ValueError(f'Invalid {name}')
    for name in ('artifact_video', 'artifact_audio'):
        if result[name].shape != (n,) or not np.isfinite(result[name]).all() or not ((result[name]>=0)&(result[name]<=1)).all():
            raise ValueError('Invalid cached artifact probability')
    for name in ('video_count', 'audio_count'):
        if not np.issubdtype(result[name].dtype, np.integer) or (result[name] < 0).any():
            raise ValueError('Invalid observation count')
    return result, manifest


def no_overlap(left, right):
    for key in ('sample_id', 'group'):
        overlap = set(left[key]) & set(right[key])
        if overlap:
            raise ValueError(f'Train/dev/test {key} overlap ({len(overlap)}); verify source groups')


def metrics(y, p):
    if not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Confirmed binary evaluation labels required')
    pred = p >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    both = len(np.unique(y)) == 2
    return dict(n=len(y), real=int((y==0).sum()), fake=int((y==1).sum()),
                roc_auc=float(roc_auc_score(y,p)) if both else None,
                average_precision=float(average_precision_score(y,p)) if both else None,
                accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)) if both else None,
                f1=float(f1_score(y,pred,zero_division=0)),
                tn=tn, fp=fp, fn=fn, tp=tp, threshold=.5)


def evaluate_arrays(data, branch, output, split, device, log):
    branch.eval()
    keep = (data['video_count']>0)&(data['audio_count']>0)&(data['labels'][:,0]>=0)
    ix = np.flatnonzero(keep)
    if not len(ix):
        raise ValueError('No common valid evaluation samples')
    ev, ea = [], []
    with torch.no_grad():
        for start in range(0, len(ix), 64):
            j = ix[start:start+64]
            result = branch(torch.from_numpy(data['video_mean'][j]).float().to(device),
                            torch.from_numpy(data['audio_mean'][j]).float().to(device))
            ev.extend(result['video_score'].cpu().tolist())
            ea.extend(result['audio_score'].cpu().tolist())
    av = torch.tensor(data['artifact_video'][ix], dtype=torch.float32)
    aa = torch.tensor(data['artifact_audio'][ix], dtype=torch.float32)
    ev, ea = torch.tensor(ev), torch.tensor(ea)
    artifact = probabilistic_or(av, aa)
    emotion = probabilistic_or(ev, ea)
    combined = HierarchicalScoreFusion()(artifact, emotion)
    y = data['labels'][ix, 0]
    report = {name: metrics(y, score.numpy()) for name, score in
              [('artifact_only', artifact), ('emotion_only', emotion), ('hierarchical', combined)]}
    report['coverage'] = dict(total=len(keep), common_valid=len(ix), excluded=int((~keep).sum()))
    pa, pf = artifact.numpy()>=.5, combined.numpy()>=.5
    report['or_changes_at_05'] = dict(additional_fake_detected=int(((y==1)&~pa&pf).sum()),
                                      additional_real_false_positives=int(((y==0)&~pa&pf).sum()))
    write_json(output/f'{split}_metrics.json', report)
    write_json(output/f'{split}_excluded.json', [dict(sample_id=str(data['sample_id'][i]),
        video_valid=int(data['video_count'][i]), audio_valid=int(data['audio_count'][i]),
        clip_label=int(data['labels'][i,0])) for i in np.flatnonzero(~keep)])
    with (output/f'{split}_predictions.jsonl').open('w') as stream:
        for k, i in enumerate(ix):
            row = dict(sample_id=str(data['sample_id'][i]), group=str(data['group'][i]),
                       clip_label=int(y[k]), Av=float(av[k]), Aa=float(aa[k]), Ev=float(ev[k]), Ea=float(ea[k]),
                       score_artifact=float(artifact[k]), score_emotion=float(emotion[k]), score_final=float(combined[k]))
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    log('evaluation_complete', split=split, metrics=report)
    return report


def train(args, log):
    tr, tm = load_cache(args.train_cache, 'train')
    dv, dm = load_cache(args.dev_cache, 'dev')
    if tm['recipe_sha256'] != dm['recipe_sha256']:
        raise ValueError('Train/dev feature recipe mismatch')
    no_overlap(tr, dv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device).startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    branch = HierarchicalEmotionBranch(hidden_dim=64, dropout=.2).to(args.device)
    selection = {}
    for modality, index in [('video', 1), ('audio', 2)]:
        head = getattr(branch, modality+'_head')
        masks = [(d[modality+'_count']>0)&(d['labels'][:,index]>=0) for d in (tr,dv)]
        tx = torch.from_numpy(tr[modality+'_mean'][masks[0]]).float()
        ty = torch.from_numpy(tr['labels'][masks[0],index]).float()
        dx = torch.from_numpy(dv[modality+'_mean'][masks[1]]).float()
        dy = dv['labels'][masks[1],index]
        if len(torch.unique(ty)) != 2 or len(np.unique(dy)) != 2:
            raise ValueError(f'{modality}: both classes needed in train and dev')
        # Clip-level statistics fitted ONLY on valid, labeled train samples.
        head.fit_standardizer(tx.to(args.device))
        pw = float((ty==0).sum()/(ty==1).sum())
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=args.device))
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),
                    batch_size=args.batch_size, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(args.seed))
        best_auc, best_state, best_epoch, stale = -1., None, 0, 0
        log('head_train_start', modality=modality, train_n=len(tx), dev_n=len(dx), pos_weight=pw)
        for epoch in range(1, args.epochs+1):
            head.train()
            loss_sum, seen = 0., 0
            for batch_index, (x,y) in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(x.to(args.device)), y.to(args.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise FloatingPointError('Missing/nonfinite head gradient')
                optimizer.step()
                loss_sum += float(loss.detach())*len(x)
                seen += len(x)
                if batch_index % 50 == 0 or batch_index == len(loader):
                    log('train_batch', modality=modality, epoch=epoch, batch=batch_index,
                        batches=len(loader), train_loss=loss_sum/seen)
            head.eval()
            with torch.no_grad():
                probs = np.concatenate([head(x.to(args.device)).sigmoid().cpu().numpy() for x in dx.split(256)])
            auc = float(roc_auc_score(dy, probs))
            if auc > best_auc:
                best_auc, best_epoch, stale = auc, epoch, 0
                best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                atomic_torch_save(args.output/f'{modality}_head_best.pt',
                                  dict(state_dict=best_state, epoch=epoch, dev_auc=auc))
            else:
                stale += 1
            log('epoch_complete', modality=modality, epoch=epoch, train_loss=loss_sum/seen,
                dev_auc=auc, best_dev_auc=best_auc, best_epoch=best_epoch, patience_used=stale)
            if stale >= args.patience:
                break
        head.load_state_dict(best_state)
        selection[modality] = dict(best_epoch=best_epoch, best_dev_auc=best_auc,
                                   pos_weight=pw, train_n=len(tx), dev_n=len(dx))
    state = dict(format='hierarchical_heads_v1', hidden_dim=64, dropout=.2,
                 state_dict={k:v.detach().cpu() for k,v in branch.state_dict().items()},
                 recipe_sha256=tm['recipe_sha256'], recipe=tm['recipe'],
                 train_cache_sha256=tm['cache_sha256'], dev_cache_sha256=dm['cache_sha256'],
                 train_ids=tr['sample_id'].tolist(), dev_ids=dv['sample_id'].tolist(),
                 train_groups=tr['group'].tolist(), dev_groups=dv['group'].tolist(),
                 selection=selection, seed=args.seed, implementation=code_hashes())
    atomic_torch_save(args.output/'emotion_heads.pt', state)
    evaluate_arrays(dv, branch, args.output, 'dev', args.device, log)
    write_json(args.output/'training_summary.json', dict(selection=selection, seed=args.seed,
           emotion_checkpoint_sha256=sha256(args.output/'emotion_heads.pt'),
           epochs_limit=args.epochs, patience=args.patience, batch_size=args.batch_size,
           learning_rate=.001, weight_decay=.0001, train_cache_sha256=tm['cache_sha256'],
           dev_cache_sha256=dm['cache_sha256'], recipe_sha256=tm['recipe_sha256']))


def atomic_torch_save(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def evaluate(args, log):
    if args.split == 'test' and not args.allow_test:
        raise ValueError('Test evaluation requires --allow-test after protocol lock')
    data, manifest = load_cache(args.cache, args.split)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state['format'] != 'hierarchical_heads_v1' or state['recipe_sha256'] != manifest['recipe_sha256']:
        raise ValueError('Checkpoint/cache recipe mismatch')
    if state['implementation'] != code_hashes():
        raise ValueError('Training/evaluation implementation changed')
    no_overlap(dict(sample_id=state['train_ids'], group=state['train_groups']), data)
    if args.split == 'test':
        no_overlap(dict(sample_id=state['dev_ids'], group=state['dev_groups']), data)
    elif manifest['cache_sha256'] != state['dev_cache_sha256']:
        raise ValueError('Use the same dev cache used for checkpoint selection')
    branch = HierarchicalEmotionBranch(hidden_dim=state['hidden_dim'], dropout=state['dropout']).to(args.device)
    branch.load_state_dict(state['state_dict'], strict=True)
    write_json(args.output/'evaluation_manifest.json', dict(split=args.split,
        checkpoint_sha256=sha256(args.checkpoint), cache_sha256=manifest['cache_sha256'],
        recipe_sha256=manifest['recipe_sha256'], threshold=.5))
    evaluate_arrays(data, branch, args.output, args.split, args.device, log)


def partition_ids(ids, count):
    if not ids or len(ids) != len(set(ids)) or count < 1:
        raise ValueError('Unique, nonempty sample IDs and positive worker count required')
    return [ids[i::count] for i in range(count)]


def multi_plan(args):
    """Read metadata on CPU only; do not instantiate four models in the launcher."""
    from src.datasets.frozen import FrozenMediaDataset
    from src.datasets.media import MediaConfig
    if not args.trusted_artifact_checkpoints:
        raise ValueError('--trusted-artifact-checkpoints is required for your existing checkpoints')
    if args.split == 'test' and not args.allow_test:
        raise ValueError('Test extraction requires --allow-test')
    config = json.loads(args.config.read_text())
    checkpoint = ROOT / config['x3d_checkpoint']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = state['provenance']['config']
    registry = ROOT / cfg['data']['split_manifest']
    quality = ROOT / cfg['data']['eligibility']
    eligibility = ROOT / json.loads(quality.read_text())['eligibility_manifest']
    selected = None
    if args.sample_ids:
        selected = [s.strip() for s in args.sample_ids.read_text().splitlines() if s.strip()]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError('Sample IDs must be unique and nonempty')
    ds = FrozenMediaDataset('FAV', args.split, MediaConfig(), sample_ids=selected,
                           eligibility_path=eligibility, task='paired', registry=registry)
    ids = [row['sample_id'] for row in ds.rows]
    if not ids:
        raise ValueError('Empty eligible cohort')
    files = [args.config, checkpoint, ROOT/config['aasist_checkpoint'], ROOT/config['acrnn_checkpoint'],
             ROOT/config.get('acrnn_source','third_party/amsdf_acrnn/ACRNN.py'), registry, quality, eligibility,
             Path.home()/'.hsemotion/enet_b0_8_best_afew.pt']
    files += [ROOT/f'src/models/{name}.py' for name in ('x3d','aasist','hsemotion','crnn')]
    files += [ROOT/'src/datasets/media.py', ROOT/'src/datasets/frozen.py']
    plan = dict(split=args.split, gpus=args.gpus, ids=ids, shards=partition_ids(ids,len(args.gpus)),
                files={str(p.resolve()):sha256(p) for p in files}, implementation=code_hashes(),
                torch_threads=args.torch_threads, format='hierarchical_multi_plan_v1')
    return plan


def merge_shards(shard_paths, expected_ids, output, split):
    """Exact ID join in frozen-manifest order. No partial cache is accepted."""
    datasets, manifests = zip(*(load_cache(p, split) for p in shard_paths))
    if len({m['recipe_sha256'] for m in manifests}) != 1:
        raise ValueError('GPU shard recipe mismatch')
    if any(set(d) != set(datasets[0]) for d in datasets):
        raise ValueError('GPU shard array schema mismatch')
    joined = {key:np.concatenate([d[key] for d in datasets]) for key in datasets[0]}
    actual = joined['sample_id'].tolist()
    if len(set(actual)) != len(actual) or set(actual) != set(expected_ids) or len(actual) != len(expected_ids):
        raise ValueError('GPU shards have duplicate, missing or unexpected IDs')
    lookup = {sid:i for i,sid in enumerate(actual)}
    order = np.asarray([lookup[sid] for sid in expected_ids])
    # Passing a stream prevents NumPy from silently appending .npz to the temporary name.
    tmp = output/'cache.npz.tmp'
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **{k:v[order] for k,v in joined.items()})
    tmp.replace(output/'cache.npz')
    manifest = dict(manifests[0])
    manifest.update(n=len(actual), cache_sha256=sha256(output/'cache.npz'),
                    shard_caches=[dict(path=str(p), sha256=m['cache_sha256']) for p,m in zip(shard_paths,manifests)])
    write_json(output/'manifest.json', manifest)
    write_json(output/'recipe.json', manifest['recipe'])
    # Per-observation audit remains in the immutable shard directories; index those files.
    write_json(output/'observations_index.json', [str(p/'observations.jsonl') for p in shard_paths])


def cache_multi(args, log):
    import os
    import subprocess
    plan = multi_plan(args)
    plan_file = args.output/'shard_plan.json'
    if plan_file.exists():
        if json.loads(plan_file.read_text()) != plan:
            raise ValueError('Resume plan/config/source/checkpoint changed; use a new output directory')
    else:
        write_json(plan_file,plan)
    if (args.output/'COMPLETED.json').exists():
        load_cache(args.output,args.split)
        log('cache_already_complete', samples=len(plan['ids']))
        return
    running, accepted = [], {}
    try:
        for shard, (gpu, ids) in enumerate(zip(args.gpus,plan['shards'])):
            if not ids:
                continue
            directory = args.output/'shards'/f'gpu_{shard}'
            directory.mkdir(parents=True,exist_ok=True)
            ids_file = directory/'sample_ids.txt'
            ids_file.write_text('\n'.join(ids)+'\n')
            completed = sorted(p for p in directory.glob('attempt_*') if (p/'COMPLETED.json').exists())
            if completed:
                data,_ = load_cache(completed[-1],args.split)
                if set(data['sample_id']) != set(ids):
                    raise ValueError('Completed shard ID mismatch')
                accepted[shard] = completed[-1]
                log('reuse_completed_shard', gpu=gpu, samples=len(ids), path=str(completed[-1]))
                continue
            attempt = directory/f'attempt_{len(list(directory.glob("attempt_*")))+1:03d}'
            command = [sys.executable,str(Path(__file__).resolve()),'cache','--config',str(args.config.resolve()),
                       '--split',args.split,'--sample-ids',str(ids_file.resolve()),
                       '--trusted-artifact-checkpoints','--device','cuda:0',
                       '--torch-threads',str(args.torch_threads),'--output',str(attempt.resolve())]
            if args.allow_test:
                command.append('--allow-test')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       OMP_NUM_THREADS=str(args.torch_threads), MKL_NUM_THREADS=str(args.torch_threads),
                       PYTHONUNBUFFERED='1')
            stream = (directory/f'{attempt.name}.log').open('w')
            try:
                process = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            except BaseException:
                stream.close()
                raise
            running.append(dict(process=process,stream=stream,shard=shard,gpu=gpu,path=attempt,total=len(ids)))
            log('gpu_worker_started', gpu=gpu, logical_device='cuda:0', pid=process.pid, samples=len(ids))
        last_report = -100.
        failures = []
        while running:
            for worker in running[:]:
                status = worker['process'].poll()
                if status is None:
                    continue
                worker['stream'].close()
                running.remove(worker)
                path = worker['path']
                if status == 0 and (path/'COMPLETED.json').exists():
                    accepted[worker['shard']] = path
                    log('gpu_worker_complete', gpu=worker['gpu'], samples=worker['total'])
                else:
                    failures.append(str(path))
                    log('gpu_worker_failed', gpu=worker['gpu'], returncode=status, path=str(path))
            if running and time.monotonic()-last_report >= 10:
                last_report = time.monotonic()
                progress = []
                done = sum(len(plan['shards'][i]) for i in accepted)
                for worker in running:
                    heartbeat = worker['path']/'heartbeat.json'
                    row = json.loads(heartbeat.read_text()) if heartbeat.exists() else {}
                    current = row.get('current',0)
                    done += current
                    progress.append(dict(gpu=worker['gpu'], current=current,total=worker['total'],
                                         stage=row.get('event','initializing')))
                log('multi_gpu_progress', current=done,total=len(plan['ids']), workers=progress)
            if running:
                time.sleep(.5)
        if failures:
            raise RuntimeError('Some shards failed; completed shards were preserved. Resume with --resume. Failed paths: '+', '.join(failures))
        paths = [accepted[i] for i,ids in enumerate(plan['shards']) if ids]
        merge_shards(paths,plan['ids'],args.output,args.split)
        log('multi_gpu_cache_complete', samples=len(plan['ids']), shards=len(paths))
    finally:
        for worker in running:
            if worker['process'].poll() is None:
                worker['process'].terminate()
        for worker in running:
            try:
                worker['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker['process'].kill()
                worker['process'].wait()
            worker['stream'].close()


def run_stage(command):
    """Keep the whole stage in one process group so interrupt cannot orphan GPU workers."""
    import os
    import signal
    import subprocess
    process = subprocess.Popen(command, start_new_session=True)
    try:
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
    except BaseException:
        # The launcher may have exited while its workers are still alive.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def run_pipeline(args, log):
    """One command: four-GPU train/dev extraction followed by lightweight head training/dev evaluation."""
    import subprocess
    plan = dict(config_sha256=sha256(args.config), gpus=args.gpus, seed=args.seed,
                epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                torch_threads=args.torch_threads, head_device=args.device, implementation=code_hashes())
    path = args.output/'pipeline_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Pipeline settings changed; use a new output directory')
    write_json(path,plan)
    for split in ('train','dev'):
        command = [sys.executable,str(Path(__file__).resolve()),'cache-multi','--config',str(args.config.resolve()),
                   '--split',split,'--gpus',*args.gpus,'--trusted-artifact-checkpoints',
                   '--torch-threads',str(args.torch_threads),'--output',str((args.output/f'cache_{split}').resolve())]
        if args.resume and (args.output/f'cache_{split}').exists():
            command.append('--resume')
        if not args.trusted_artifact_checkpoints:
            raise ValueError('--trusted-artifact-checkpoints is required')
        log('pipeline_stage', stage=f'cache_{split}')
        # Inherit stdout/stderr so child progress is immediately visible in the terminal.
        run_stage(command)
    runs = args.output/'head_runs'
    runs.mkdir(exist_ok=True)
    completed = sorted(p for p in runs.glob('attempt_*') if (p/'COMPLETED.json').exists())
    if completed:
        selected = completed[-1]
        summary = json.loads((selected/'training_summary.json').read_text())
        for split in ('train','dev'):
            _,manifest = load_cache(args.output/f'cache_{split}',split)
            if summary[f'{split}_cache_sha256'] != manifest['cache_sha256']:
                raise ValueError('Completed head training uses different caches')
        log('reuse_completed_training', path=str(selected))
    else:
        selected = runs/f'attempt_{len(list(runs.glob("attempt_*")))+1:03d}'
        command = [sys.executable,str(Path(__file__).resolve()),'train',
                   '--train-cache',str((args.output/'cache_train').resolve()),
                   '--dev-cache',str((args.output/'cache_dev').resolve()),'--seed',str(args.seed),
                   '--epochs',str(args.epochs),'--patience',str(args.patience),'--batch-size',str(args.batch_size),
                   '--device',args.device,'--torch-threads',str(args.torch_threads),'--output',str(selected.resolve())]
        log('pipeline_stage',stage='train_heads_and_evaluate_dev')
        run_stage(command)
    write_json(args.output/'pipeline_result.json',dict(head_run=str(selected.resolve()),
        checkpoint=str((selected/'emotion_heads.pt').resolve()),dev_metrics=str((selected/'dev_metrics.json').resolve())))
    log('pipeline_complete',head_run=str(selected),test_access=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cache', help='Frozen backbone inference; one sample at a time')
    c.add_argument('--config', type=Path, required=True)
    c.add_argument('--split', choices=['train','dev','test'], required=True)
    c.add_argument('--sample-ids', type=Path)
    c.add_argument('--trusted-artifact-checkpoints', action='store_true')
    c.add_argument('--allow-test', action='store_true')
    t = sub.add_parser('train', help='Train the two emotion MLP heads only; never reads test')
    t.add_argument('--train-cache', type=Path, required=True)
    t.add_argument('--dev-cache', type=Path, required=True)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--epochs', type=int, default=30)
    t.add_argument('--patience', type=int, default=5)
    t.add_argument('--batch-size', type=int, default=64)
    e = sub.add_parser('evaluate')
    e.add_argument('--cache', type=Path, required=True)
    e.add_argument('--checkpoint', type=Path, required=True)
    e.add_argument('--split', choices=['dev','test'], required=True)
    e.add_argument('--allow-test', action='store_true')
    m = sub.add_parser('cache-multi', help='Independent GPU workers, verified shard merge and explicit resume')
    m.add_argument('--split', choices=['train','dev','test'], required=True)
    m.add_argument('--sample-ids', type=Path)
    m.add_argument('--allow-test', action='store_true')
    r = sub.add_parser('run', help='4-GPU train/dev cache -> emotion head training -> dev evaluation')
    r.add_argument('--seed', type=int, default=42)
    r.add_argument('--epochs', type=int, default=30)
    r.add_argument('--patience', type=int, default=5)
    r.add_argument('--batch-size', type=int, default=64)
    for command in (m,r):
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--gpus', nargs='+', default=['0','1','2','3'],
                             help='CUDA device selectors; GPU UUIDs are also accepted')
        command.add_argument('--trusted-artifact-checkpoints', action='store_true')
        command.add_argument('--resume', action='store_true', help='Reuse completed shards; restart only unfinished shards')
    for command in (c,t,e,m,r):
        command.add_argument('--device', default='cpu')
        command.add_argument('--torch-threads', type=int, default=2)
        command.add_argument('--output', type=Path, required=True, help='New directory; existing output is never overwritten')
    args = p.parse_args()
    if args.command in ('train','run') and min(args.epochs,args.patience,args.batch_size)<=0:
        p.error('Positive training values required')
    if args.torch_threads < 1:
        p.error('--torch-threads must be positive')
    if args.command in ('cache-multi','run') and (len(args.gpus) != len(set(args.gpus)) or any(',' in g for g in args.gpus)):
        p.error('--gpus must list distinct GPU selectors separated by spaces')
    return args


def main():
    args = parse_args()
    resume = getattr(args, 'resume', False)
    if resume and args.output.exists() and not (args.output/'run_manifest.json').exists():
        raise ValueError('Not a previous experiment directory')
    args.output.mkdir(parents=True, exist_ok=resume)
    torch.set_num_threads(args.torch_threads)
    log = Logger(args.output)
    manifest_path = args.output/('resume_'+str(time.time_ns())+'.json' if resume else 'run_manifest.json')
    if not (args.output/'run_manifest.json').exists():
        manifest_path = args.output/'run_manifest.json'
    write_json(manifest_path, dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        implementation=code_hashes(), versions={'torch':str(torch.__version__), 'numpy':np.__version__},
        policy='frozen backbones; modality BCE heads; fixed OR; threshold 0.5; no automatic retry'))
    try:
        log('start', command=args.command)
        {'cache':cache, 'train':train, 'evaluate':evaluate,
         'cache-multi':cache_multi, 'run':run_pipeline}[args.command](args,log)
        if (args.output/'FAILED.json').exists():
            (args.output/'FAILED.json').rename(args.output/f'FAILED_previous_{time.time_ns()}.json')
        write_json(args.output/'COMPLETED.json', {'command':args.command, 'status':'complete'})
    except BaseException as error:
        (args.output/'error.txt').write_text(traceback.format_exc())
        write_json(args.output/'FAILED.json', {'type':type(error).__name__, 'error':str(error)})
        log('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
