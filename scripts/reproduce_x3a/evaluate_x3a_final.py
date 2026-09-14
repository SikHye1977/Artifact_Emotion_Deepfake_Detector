"""Freeze dev choices, then run one final FAV test; no training or DDP."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from scripts.reproduce_x3a.train_x3a import prepare, digest, atomic_json, metrics, emit, validate_predictions
from src.datasets.frozen import FrozenMediaDataset

NAMES = ('x3d', 'aasist')
SCOPES = ('paper_x3d', 'paper_aasist', 'paper_x3a', 'research_x3d', 'research_aasist')
LABEL = dict(paper_x3d=1, paper_aasist=2, paper_x3a=0, research_x3d=1, research_aasist=2)


def category(labels):
    _, v, a = labels
    return None if v is None or a is None else ('FV' if v else 'RV') + ('FA' if a else 'RA')


def select(rows, scope):
    column = LABEL[scope]
    prob = 'or_probability' if column == 0 else ('video_probability' if column == 1 else 'audio_probability')
    keep = []
    for r in rows:
        if r['labels'][column] is None or r.get(prob) is None:
            continue
        if scope == 'paper_x3d' and r['category'] not in ('RVRA', 'FVRA', 'FVFA'):
            continue
        if scope == 'paper_aasist' and r['category'] not in ('RVRA', 'RVFA', 'FVFA'):
            continue
        keep.append((r['labels'][column], r[prob]))
    if {y for y, p in keep} != {0, 1}:
        raise ValueError(f'{scope}: both classes required')
    return np.asarray([y for y, p in keep]), np.asarray([p for y, p in keep])


def threshold_metrics(y, p, threshold):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
    pred = p >= threshold
    return dict(threshold=float(threshold), accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
                mcc=float(matthews_corrcoef(y, pred)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def choose_threshold(y, p):
    # Exact >= cutoffs, including all-positive/all-negative; ties: closest to .5, then lower.
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    scores = (tpr + 1 - fpr) / 2
    candidates = [(float(s), float(t)) for s, t in zip(scores, thresholds) if 0 <= t <= 1]
    candidates += [(threshold_metrics(y, p, t)['balanced_accuracy'], t) for t in (0., .5, 1.)]
    return min(candidates, key=lambda x: (-x[0], abs(x[1] - .5), x[1]))[1]


def join_predictions(predictions, datasets, paired_ids, split):
    merged = {}
    for name, column, key in [('x3d', 1, 'video_probability'), ('aasist', 2, 'audio_probability')]:
        source = {r['sample_id']: r for r in datasets[name].rows}
        validate_predictions(predictions[name], list(source))
        for r in predictions[name]:
            s = source[r['sample_id']]
            if r.get('split') != split or r['label'] != s['labels'][column]:
                raise ValueError('Prediction split/label mismatch')
            p = r['fake_probability']
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Nonfinite/out-of-range probability')
            row = merged.setdefault(r['sample_id'], dict(sample_id=r['sample_id'], split=split,
                labels=s['labels'], category=category(s['labels']),
                video_probability=None, audio_probability=None, or_probability=None))
            if row['labels'] != s['labels']:
                raise ValueError('Modality metadata mismatch')
            row[key] = p
            row[name + '_logit'] = r['logit']
    for sid in paired_ids:
        if sid not in merged or any(merged[sid][k] is None for k in ('video_probability', 'audio_probability')):
            raise ValueError('Missing paired prediction')
        row = merged[sid]
        row['or_probability'] = 1 - (1 - row['video_probability']) * (1 - row['audio_probability'])
    return [merged[s] for s in sorted(merged)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='x3a_4gpu_001')
    parser.add_argument('--execute-test', action='store_true', help='Without this flag: freeze dev only')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('Invalid run ID')
    base = Path('experiments/x3a_reproduction/fav_indomain_v1/unified_4gpu') / args.run_id
    report, ckpts = ROOT/'reports'/base, ROOT/'checkpoints'/base
    output = report/'final_test'
    if (output/'STARTED.json').exists():
        raise FileExistsError('Final test already attempted; preserve outputs and investigate instead of rerunning')
    prepared, hashes, dev_predictions, deps = {}, {}, {}, {}
    for name, task in [('x3d', 'video'), ('aasist', 'audio')]:
        completion = json.loads((report/name/'COMPLETED.json').read_text())
        if completion['epochs'] != 20:
            raise ValueError('20 epochs required')
        checkpoint = ckpts/name/'best.pt'
        hashes[name] = digest(checkpoint)
        if hashes[name] != completion['sha256']['best.pt']:
            raise ValueError('Best checkpoint hash mismatch')
        config_path = ROOT/f'configs/experiments/x3a_reproduction/fav_indomain_v1/{name}_4gpu.yaml'
        prepared[name] = prepare(config_path, ROOT, task, torch.device('cpu'))
        config, _, dev, _, _, current = prepared[name]
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        old = state['provenance']
        for key in ('config', 'config_sha256', 'split_manifest_sha256', 'eligibility_sha256',
                    'registry_sha256', 'input_recipe_sha256', 'model_source_sha256', 'weight', 'runtime'):
            if old[key] != current[key]:
                raise ValueError(f'{name}: provenance mismatch: {key}')
        positions = [r['position'] for r in state['rank_states']]
        if not all(p['epoch'] == completion['best_epoch'] and p['epoch_complete'] for p in positions):
            raise ValueError('Best checkpoint epoch mismatch')
        path = report/name/'best_dev.jsonl'
        dev_predictions[name] = [json.loads(s) for s in path.read_text().splitlines()]
        if {r['checkpoint_epoch'] for r in dev_predictions[name]} != {completion['best_epoch']}:
            raise ValueError('Dev prediction checkpoint mismatch')
        deps[name] = dict(checkpoint_sha256=hashes[name], dev_prediction_sha256=digest(path),
                          best_epoch=completion['best_epoch'], provenance=old)
        del state
    config = prepared['aasist'][0]
    quality = json.loads((ROOT/config['data']['eligibility']).read_text())
    kwargs = dict(eligibility_path=ROOT/quality['eligibility_manifest'], registry=ROOT/config['data']['split_manifest'])
    paired_dev = FrozenMediaDataset('FAV', 'dev', task='paired', **kwargs)
    dev_rows = join_predictions(dev_predictions, {n: prepared[n][2] for n in NAMES},
                                {r['sample_id'] for r in paired_dev.rows if r['labels'][0] is not None}, 'dev')
    thresholds = {}
    for scope in SCOPES:
        y, p = select(dev_rows, scope)
        thresholds[scope] = choose_threshold(y, p)
    frozen = dict(version=1, run_id=args.run_id, checkpoints=deps, thresholds=thresholds,
        fixed_threshold=.5, class_mapping={'real': 0, 'fake': 1}, fake_class_index=1,
        evaluator_sha256=digest(__file__), trainer_sha256=digest(ROOT/'scripts/reproduce_x3a/train_x3a.py'),
        selection='dev balanced accuracy; ties closest to .5 then lower',
        inference=dict(batch_size=1, x3d_amp=True, aasist_amp=False),
        scope=list(SCOPES), test_used_for_selection=False)
    output.mkdir(exist_ok=True)
    freeze_path = output/'frozen_settings.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text()) != frozen:
            raise ValueError('Frozen settings changed; test refused')
    else:
        atomic_json(freeze_path, frozen)
    emit('FROZEN', path=freeze_path, sha256=digest(freeze_path), thresholds=thresholds)
    if not args.execute_test:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; test not started')
    # Exclusive claim before any test split is constructed. Never automatically retry failed test.
    with (output/'STARTED.json').open('x') as f:
        json.dump(dict(started=time.time(), frozen_sha256=digest(freeze_path)), f)
    started = time.monotonic()
    monitor = None
    try:
        from src.utils.gpu_monitor import GPUMonitor, ThermalPolicy
        monitor = GPUMonitor(output, 'gpu_telemetry.jsonl', interval_seconds=10,
                             policy=ThermalPolicy(80, 88, 1, 10), stop_enabled=False)
        monitor.start()
        local = yaml.safe_load((ROOT/'configs/local_paths.yaml').read_text())
        predictions, datasets = {}, {}
        for name, task in [('x3d', 'video'), ('aasist', 'audio')]:
            config, _, dev, factory, collate, _ = prepared[name]
            ds = FrozenMediaDataset('FAV', 'test', config=dev.config, task=task, **kwargs)
            column = 1 if task == 'video' else 2
            ds.rows = [r for r in ds.rows if r['labels'][column] is not None]
            for row in ds.rows:
                row['path'] = str(Path(local['fav_root'])/row['sample_id'].split(':', 1)[1])
            datasets[name] = ds
            model = factory()
            if digest(ckpts/name/'best.pt') != hashes[name]:
                raise ValueError('Checkpoint changed after freeze')
            state = torch.load(ckpts/name/'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            del state
            model.to('cuda').eval()
            rows = []
            loader = DataLoader(ds, batch_size=1, collate_fn=collate, num_workers=0)
            with torch.inference_mode(), (output/f'{name}_predictions.jsonl').open('x') as stream:
                for i, batch in enumerate(loader):
                    x = batch[task].to('cuda')
                    expected = (3, 128, 256, 256) if task == 'video' else (284672,)
                    if tuple(x.shape[1:]) != expected or not torch.isfinite(x).all():
                        raise ValueError('Invalid media input')
                    if task == 'audio' and (model.fake_class_index != 1 or (batch['audio_lengths'] > 284672).any()):
                        raise ValueError('Audio mapping/truncation mismatch')
                    with torch.autocast('cuda', dtype=torch.float16, enabled=(task == 'video')):
                        result = model(x)
                    if not torch.isfinite(result.logits).all():
                        raise ValueError('Nonfinite model output')
                    row = dict(sample_id=batch['sample_id'][0], split='test',
                        label=int(batch['labels'][0, column]), logit=result.logits.float().cpu().tolist()[0],
                        fake_probability=float(result.fake_probability[0]))
                    rows.append(row)
                    stream.write(json.dumps(row)+'\n')
                    if i % 10 == 0 or i+1 == len(loader):
                        stream.flush()
                        emit('TEST', model=name, batch=i+1, total=len(loader), seconds=time.monotonic()-started)
            validate_predictions(rows, [r['sample_id'] for r in ds.rows])
            predictions[name] = rows
            del model
            torch.cuda.empty_cache()
        paired_test = FrozenMediaDataset('FAV', 'test', task='paired', **kwargs)
        rows = join_predictions(predictions, datasets,
            {r['sample_id'] for r in paired_test.rows if r['labels'][0] is not None}, 'test')
        with (output/'predictions.jsonl').open('x') as f:
            for row in rows:
                row.update(clip_label=row['labels'][0], video_label=row['labels'][1], audio_label=row['labels'][2],
                    checkpoint_sha256=hashes, config_sha256={n: deps[n]['provenance']['config_sha256'] for n in NAMES},
                    frozen_settings_sha256=digest(freeze_path))
                f.write(json.dumps(row)+'\n')
        scores = {}
        for scope in SCOPES:
            y, p = select(rows, scope)
            scores[scope] = dict(samples=len(y), metrics=metrics(y,p),
                dev_selected=threshold_metrics(y,p,thresholds[scope]))
        atomic_json(output/'summary.json', dict(results=scores, seconds=time.monotonic()-started,
            frozen_sha256=digest(freeze_path), checkpoints=hashes,
            limitation='paper-spec reimplementation: custom frozen split; unpublished optimizer/epoch/MLP/padding details'))
        atomic_json(output/'COMPLETED.json', dict(completed=True, seconds=time.monotonic()-started,
            summary_sha256=digest(output/'summary.json')))
        emit('TEST_COMPLETED', path=output/'summary.json')
    except BaseException:
        atomic_json(output/'FAILED.json', dict(traceback=traceback.format_exc(), completed=False))
        raise
    finally:
        if monitor is not None:
            monitor.close()


if __name__ == '__main__':
    main()
