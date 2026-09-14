"""Unified FAV X3A training: single GPU or four-GPU DDP; never evaluates test.

All training orchestration, AMP updates, checkpoint/resume and progress reporting
live here. Model wrappers and frozen-media/metric utilities remain reusable.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from src.evaluation.metrics import binary_metrics
from src.evaluation.evaluate import validate_predictions


def emit(stage, **fields):
    print(time.strftime('%H:%M:%S'), stage,
          ' '.join(f'{key}={value}' for key, value in fields.items()), flush=True)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def epoch_draws(labels, count, seed, epoch):
    """Exact 1:1 replacement draws; independent epoch generator, no rank/sharding."""
    pools = [torch.tensor([i for i, y in enumerate(labels) if y == cls]) for cls in (0, 1)]
    if not all(len(pool) for pool in pools):
        raise ValueError('Both training classes are required')
    generator = torch.Generator().manual_seed(seed + epoch)
    sizes = (count // 2, count - count // 2)
    draws = torch.cat([pool[torch.randint(len(pool), (n,), generator=generator)]
                       for pool, n in zip(pools, sizes)])
    return draws[torch.randperm(count, generator=generator)].tolist()


def batch_forward(model, batch, device, task):
    if set(batch['split']) - {'train', 'dev'}:
        raise ValueError('Test/unknown split forbidden')
    column = 1 if task == 'video' else 2
    mask = batch['label_mask'][:, column].bool()
    if not mask.any():
        raise ValueError('Batch has no supervised labels')
    y = batch['labels'][mask, column].to(device)
    x = batch[task][mask].to(device)
    expected = (3, 128, 256, 256) if task == 'video' else (284672,)
    if tuple(x.shape[1:]) != expected:
        raise ValueError(f'Invalid {task} shape: {tuple(x.shape)}')
    if task == 'audio':
        base_model = getattr(model, 'module', model)
        if base_model.fake_class_index != 1 or (batch['audio_lengths'] > 284672).any():
            raise ValueError('AASIST mapping/truncation mismatch')
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite input')
    out = model(x)
    if not torch.isfinite(out.logits).all() or not torch.isfinite(out.fake_probability).all():
        raise FloatingPointError('Nonfinite model output')
    ids = [sid for sid, keep in zip(batch['sample_id'], mask.tolist()) if keep]
    return out, y, ids


def make_optimizer(model, config, device):
    t = config['training']; o = t['optimizer']
    cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[o['name']]
    optimizer = cls(model.parameters(), lr=o['learning_rate'], weight_decay=o['weight_decay'])
    if t['scheduler']['name'] != 'cosine':
        raise ValueError('Only configured cosine scheduler is supported')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t['epochs'])
    scaler = torch.amp.GradScaler(device.type, enabled=t['mixed_precision'],
                                 init_scale=t.get('amp_initial_scale', 16384))
    return optimizer, scheduler, scaler


def optimizer_update(model, optimizer, scaler, config, consecutive, parallel=None):
    old = float(scaler.get_scale())
    scaler.unscale_(optimizer)
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing = [n for n, p in parameters.items() if p.grad is None]
    nonfinite = [n for n, p in parameters.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
    required = set(config.get('required_gradient_parameters', []))
    invalid = bool(required - parameters.keys() or required.intersection(missing) or len(missing) == len(parameters))
    if parallel:
        parallel.check('Required/no gradient: '+str(missing) if invalid else None)
    elif invalid:
        raise FloatingPointError(f'Required/no gradient: {missing}')
    overflow = bool(nonfinite)
    if parallel:
        overflow = any(parallel.collect(overflow))
    if overflow and not scaler.is_enabled():
        raise FloatingPointError(f'FP32 nonfinite gradients: {nonfinite}')
    if not overflow and config.get('gradient_clip_norm') is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
    if overflow and parallel:
        # All ranks skip together, including ranks with locally finite gradients.
        # Public explicit scale update resets GradScaler optimizer bookkeeping.
        scaler.update(new_scale=old * scaler.get_backoff_factor())
    else:
        scaler.step(optimizer); scaler.update()
    bad = not overflow and any(not torch.isfinite(p).all() for p in parameters.values())
    if parallel:
        parallel.check('Nonfinite parameter after optimizer update' if bad else None)
    elif bad:
        raise FloatingPointError('Nonfinite parameter after optimizer update')
    consecutive = consecutive + 1 if overflow else 0
    record = dict(overflow=overflow, step_performed=not overflow,
                  previous_scale=old, new_scale=float(scaler.get_scale()),
                  missing_gradient=missing, nonfinite_gradient=nonfinite, consecutive=consecutive)
    if consecutive >= config.get('max_consecutive_overflows', 5) or record['new_scale'] < config.get('amp_min_scale', 1):
        raise FloatingPointError(f'Repeated overflow or minimum scale: {record}')
    return record


def capture_rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def save_state(path, model, optimizer, scheduler, scaler, position, provenance, device):
    """Saved only at optimizer boundaries, or epoch completion after dev/scheduler."""
    payload = dict(format='x3a_single_v1', model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), position=dict(position),
                   rng=capture_rng(device), provenance=provenance)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    emit('CHECKPOINT', file=path, epoch=position['epoch'], offset=position['offset'],
         successful=position['successful'])


def load_state(path, model, optimizer, scheduler, scaler, provenance, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format') != 'x3a_single_v1':
        raise ValueError('Legacy/DDP checkpoint is not an exact single-GPU resume; start a new run')
    if state['provenance'] != provenance:
        raise ValueError('Resume config/data/source/weight/device provenance mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler']); restore_rng(state['rng'], device)
    return state['position']


@contextmanager
def stop_requests():
    """Finish the current update and atomically save on Ctrl+C; do not start dev."""
    request = {'signal': None}
    previous = {}
    def handler(number, frame):
        request['signal'] = number
        emit('STOP_REQUESTED', signal=number, action='save at next optimizer boundary')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[sig] = signal.signal(sig, handler)
    try:
        yield request
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


def metrics(labels, probabilities):
    from sklearn.metrics import roc_curve, log_loss
    result = binary_metrics(labels, probabilities)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    index = np.argmin(np.abs(fpr - (1 - tpr)))
    result.update(eer=float((fpr[index] + 1 - tpr[index]) / 2),
                  log_loss=float(log_loss(labels, probabilities, labels=[0, 1])))
    return result


def better(candidate, best):
    return best is None or candidate['roc_auc'] > best['roc_auc'] or (
        candidate['roc_auc'] == best['roc_auc'] and candidate['eer'] < best['eer'])


@torch.no_grad()
def dev_evaluate(model, dataset, collate, task, device, amp, batch_size, progress, stop):
    model.eval(); records = []
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    for i, batch in enumerate(loader):
        if stop['signal'] is not None:
            raise InterruptedError('Stop requested during dev')
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out, labels, ids = batch_forward(model, batch, device, task)
        records.extend(dict(sample_id=s, split='dev', label=y, logit=z, fake_probability=p)
                       for s, y, z, p in zip(ids, labels.tolist(), out.logits.float().tolist(), out.fake_probability.float().tolist()))
        progress('dev', i + 1, len(loader), samples=len(records))
    validate_predictions(records, [r['sample_id'] for r in dataset.rows])
    return records, metrics([r['label'] for r in records], [r['fake_probability'] for r in records])
def prepare(config_path, project_root, task, device):
    import subprocess
    from functools import partial
    import yaml
    from src.utils.config import load_experiment,sha256,resolve_sampler_policy,sampler_policy_sha256,validate_sampler_policy
    from src.utils.checkpointing import checkpoint_metadata
    from src.utils.reproducibility import seed_everything
    from src.datasets.frozen import FrozenMediaDataset
    from src.datasets.media import MediaConfig,collate_media
    root=Path(project_root).resolve();config,_=load_experiment(config_path,root)
    name='x3d' if task=='video' else 'aasist';t=config['training']
    required_batch=(2, t['gradient_accumulation_steps'], True) if task=='video' else (4, t['gradient_accumulation_steps'], False)
    if (t['batch_size'],t['mixed_precision'])!=(t['per_device_batch_size'],required_batch[2]):
        raise ValueError('Training does not match input_recipe batch/precision')
    sampler_policy=resolve_sampler_policy(config)
    validate_sampler_policy(sampler_policy,task=task,expected_world_size=config.get('distributed',{}).get('world_size',1))
    if config['checkpoint_selection']!={'split':'dev','metric':'roc_auc','mode':'max'}:
        raise ValueError('Expected best dev ROC-AUC selection')
    world=config.get('distributed',{}).get('world_size',1)
    if t['global_effective_batch_size'] != world*t['batch_size']*t['gradient_accumulation_steps']:raise ValueError('Effective batch mismatch')
    seed_everything(config['experiment']['seed']);torch.set_num_threads(2)
    quality=json.loads((root/config['data']['eligibility']).read_text())
    registry=root/config['data']['split_manifest'];eligibility=root/quality['eligibility_manifest']
    if sha256(registry)!=quality['fixed_manifest_registry_sha256'] or sha256(eligibility)!=quality['eligibility_manifest_sha256']:
        raise ValueError('Frozen data registry/eligibility hash mismatch')
    entry=json.loads(registry.read_text())['datasets']['FAV']
    if sha256(root/entry['manifest'])!=entry['sha256']:raise ValueError('Manifest hash mismatch')
    recipe=root/'reports/phase1_preflight/input_recipe.json'
    recipe_data=json.loads(recipe.read_text())
    if recipe_data['audio']['target_num_samples']!=284672:raise ValueError('Recipe mismatch')
    local=yaml.safe_load((root/'configs/local_paths.yaml').read_text())
    source_hashes={str(p.relative_to(root)):sha256(p) for p in sorted((root/'src').rglob('*.py'))}
    weight_info={'initialization':'random','sha256':None}
    if task=='video':
        from src.models.x3d import X3DDeepfakeClassifier
        from pytorchvideo.models.hub.x3d import checkpoint_paths
        import pytorchvideo
        if config['model']['name']!='x3d_m' or not config['model']['pretrained']:raise ValueError('Expected pretrained X3D-M')
        url=checkpoint_paths['x3d_m'];cache=root/'checkpoints/pretrained';cache.mkdir(parents=True,exist_ok=True)
        state=torch.hub.load_state_dict_from_url(url,model_dir=str(cache),map_location='cpu',progress=True)
        weight_info={'url':url,'sha256':sha256(cache/'X3D_M.pyth'),'initialization':'Kinetics-400'}
        for p in sorted(Path(pytorchvideo.__file__).parent.rglob('*.py')):
            source_hashes['pytorchvideo/'+str(p.relative_to(Path(pytorchvideo.__file__).parent))]=sha256(p)
        def factory():
            model=X3DDeepfakeClassifier('x3d_m',pretrained=False,freeze_backbone=config['model']['freeze_backbone'])
            weights={k:v for k,v in state['model_state'].items() if not k.startswith('blocks.5.proj.')}
            incompatible=model.backbone.load_state_dict(weights,strict=False)
            if set(incompatible.missing_keys)!={'blocks.5.proj.weight','blocks.5.proj.bias'} or incompatible.unexpected_keys:
                raise ValueError('Unexpected pretrained architecture mismatch')
            return model
        v=config['video'];media=MediaConfig(frames=v['num_frames'],crop_size=v['crop_size'],resize_short=v['resize_short_side'],mean=tuple(v['normalization']['mean']),std=tuple(v['normalization']['std']))
        collate=collate_media
    else:
        from src.models.aasist import build_aasist
        m=config['model'];assert m['class_mapping']=={'real':0,'fake':1} and m['fake_class_index']==1
        if config['audio']['target_num_samples']!=284672:raise ValueError('AASIST recipe mismatch')
        vendor=root/config['third_party']['path']
        commit=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
        if commit!=config['third_party']['commit'] or subprocess.check_output(['git','-C',str(vendor),'status','--porcelain'],text=True).strip():
            raise ValueError('Vendor commit or clean tree mismatch')
        source_hashes['official_aasist']=sha256(vendor/'models/AASIST.py')
        def factory():return build_aasist(vendor,m['official_config'],fake_class_index=1)
        media=MediaConfig(frames=1,crop_size=1,resize_short=1,audio_target_num_samples=284672)
        collate=partial(collate_media,audio_target_num_samples=284672)
    provenance=dict(metadata=checkpoint_metadata(config),config=config,config_sha256=sha256(config_path),
        split_manifest_sha256=entry['sha256'],eligibility_sha256=sha256(eligibility),registry_sha256=sha256(registry),
        input_recipe_sha256=sha256(recipe),model_source_sha256=source_hashes,weight=weight_info,
        runtime={'torch':torch.__version__,'cuda':torch.version.cuda})
    provenance['sampler_policy']=resolve_sampler_policy(config)
    provenance['sampler_policy_sha256']=sampler_policy_sha256(provenance['sampler_policy'])
    provenance['unified_source_sha256']=sha256(__file__)
    label_index=1 if task=='video' else 2
    def dataset(split):
        if split not in ('train','dev'):raise ValueError('Only train/dev are allowed')
        ds=FrozenMediaDataset('FAV',split,media,eligibility_path=eligibility,task=task,registry=registry)
        ds.rows=[row for row in ds.rows if row['labels'][label_index] is not None]
        # Relocate by dataset-relative category path without modifying the manifest.
        for row in ds.rows:
            relative=row['sample_id'].split(':',1)[1]
            row['path']=str(Path(local['fav_root'])/relative)
        return ds
    train=dataset('train')
    return config,train,dataset('dev'),factory,collate,provenance


def train_model(name, args, run_dir, checkpoint_dir, device):
    config_path = ROOT / f'configs/experiments/x3a_reproduction/fav_indomain_v1/{name}_single_gpu.yaml'
    task = 'video' if name == 'x3d' else 'audio'
    report = run_dir / name; outdir = checkpoint_dir / name
    report.mkdir(exist_ok=True); outdir.mkdir(exist_ok=True)
    emit('PREPARE', model=name, config=config_path)
    config, train, dev, factory, collate, provenance = prepare(config_path, ROOT, task, device)
    t = config['training']; batch_size = t['batch_size']; accum = t['gradient_accumulation_steps']
    provenance['execution'] = dict(mode='single_gpu', visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                   effective_batch=batch_size * accum)
    position = dict(epoch=0, offset=0, epoch_complete=False, successful=0, attempts=0,
                    overflows=0, consecutive=0, best=None, best_epoch=None,
                    loss_sum=0., loss_count=0, labels={'0': 0, '1': 0})
    model = factory().to(device)
    optimizer, scheduler, scaler = make_optimizer(model, config, device)
    resume = outdir / 'recovery.pt'
    resume_stage = args.resume and not (args.model == 'all' and not any(report.iterdir()) and not any(outdir.iterdir()))
    if resume_stage:
        if not resume.exists():
            raise FileNotFoundError(f'No single-GPU recovery: {resume}')
        position = load_state(resume, model, optimizer, scheduler, scaler, provenance, device)
        emit('RESUME', sha256=digest(resume), **position)
    atomic_json(report / 'run_manifest.json', provenance)
    column = 1 if task == 'video' else 2
    labels = [r['labels'][column] for r in train.rows]
    started = time.monotonic(); last_print = [0.]
    def progress(stage, current, total, **fields):
        now = time.monotonic()
        if current not in (1, total) and now - last_print[0] < args.log_seconds:
            return
        last_print[0] = now
        emit(stage.upper(), model=name, epoch=f"{position['epoch'] + 1}/{t['epochs']}",
             batch=f'{current}/{total}', percent=f'{100*current/max(total,1):.1f}',
             successful=position['successful'], attempts=position['attempts'], scale=scaler.get_scale(), **fields)
        atomic_json(report / 'run_status.json', dict(status=stage, position=position,
            scale=scaler.get_scale(), batch=current, total_batches=total,
            updated_at=time.time(), test_accessed=False, completed=False))
    with stop_requests() as stop:
        try:
            if not resume_stage:
                save_state(resume, model, optimizer, scheduler, scaler, position, provenance, device)
            if position['epoch_complete']:
                position.update(epoch=position['epoch'] + 1, offset=0, epoch_complete=False,
                                loss_sum=0., loss_count=0, labels={'0': 0, '1': 0})
            for epoch in range(position['epoch'], t['epochs']):
                position['epoch'] = epoch
                draws = epoch_draws(labels, t['sampler']['global_epoch_samples'], config['experiment']['seed'], epoch)
                offset = position['offset']
                if not 0 <= offset <= len(draws):
                    raise ValueError('Invalid sampler cursor')
                remaining = draws[offset:]
                # Dedicated loader generator prevents iterator creation on resume
                # from consuming the model dropout RNG.
                loader = DataLoader(Subset(train, remaining), batch_size=batch_size, collate_fn=collate,
                                    num_workers=0, generator=torch.Generator().manual_seed(42 + epoch))
                model.train(); optimizer.zero_grad(set_to_none=True)
                pending = 0
                emit('TRAIN_START', model=name, epoch=epoch, sampler_offset=offset,
                     batches=len(loader), successful=position['successful'])
                for i, batch in enumerate(loader):
                    with torch.autocast(device.type, dtype=torch.float16, enabled=t['mixed_precision']):
                        output, y, ids = batch_forward(model, batch, device, task)
                        loss = (torch.nn.functional.binary_cross_entropy_with_logits(output.logits, y.float())
                                if task == 'video' else torch.nn.functional.cross_entropy(output.logits, y.long()))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite loss')
                    group_start = (i // accum) * accum * batch_size
                    group_samples = min(accum * batch_size, len(remaining) - group_start)
                    scaler.scale(loss * (len(y) / group_samples)).backward()
                    pending += len(y)
                    position['loss_sum'] += float(loss.detach()) * len(y)
                    position['loss_count'] += len(y)
                    for y_value in y.tolist():
                        position['labels'][str(y_value)] += 1
                    if (i + 1) % accum == 0 or i + 1 == len(loader):
                        record = optimizer_update(model, optimizer, scaler, t, position['consecutive'])
                        position['attempts'] += 1
                        position['successful'] += int(record['step_performed'])
                        position['overflows'] += int(record['overflow'])
                        position['consecutive'] = record['consecutive']
                        position['offset'] += pending; pending = 0
                        optimizer.zero_grad(set_to_none=True)
                        with (report / 'updates.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(record, epoch=epoch, offset=position['offset'],
                                successful=position['successful'], attempt=position['attempts'], loss=float(loss.detach()))) + '\n')
                        if record['overflow'] or record['missing_gradient']:
                            emit('GRADIENT', **record)
                        if (record['step_performed'] and position['successful'] % t.get('recovery_interval_steps', 200) == 0) or stop['signal']:
                            save_state(resume, model, optimizer, scheduler, scaler, position, provenance, device)
                        if stop['signal']:
                            raise InterruptedError('User requested stop; checkpoint saved')
                    progress('train', i + 1, len(loader), loss=f'{float(loss.detach()):.6f}',
                             sampler_offset=position['offset'])
                # End-of-train recovery explicitly has epoch_complete=False:
                # resume repeats only dev, never the completed train draws.
                save_state(resume, model, optimizer, scheduler, scaler, position, provenance, device)
                records, result = dev_evaluate(model, dev, collate, task, device, t['mixed_precision'], batch_size, progress, stop)
                improved = better(result, position['best'])
                if improved:
                    position['best'] = result; position['best_epoch'] = epoch
                # The cosine scheduler advances once per completed epoch, never on overflow.
                scheduler.step(); position['epoch_complete'] = True
                for row in records:
                    row['checkpoint_epoch'] = epoch
                predfile = report / f'dev_{epoch:03d}.jsonl'
                predfile.write_text(''.join(json.dumps(row) + '\n' for row in records))
                if improved:
                    (report / 'best_dev.jsonl').write_text(predfile.read_text())
                    save_state(outdir / 'best.pt', model, optimizer, scheduler, scaler, position, provenance, device)
                save_state(outdir / 'last.pt', model, optimizer, scheduler, scaler, position, provenance, device)
                save_state(resume, model, optimizer, scheduler, scaler, position, provenance, device)
                atomic_json(report / f'epoch_{epoch:03d}.json', dict(epoch=epoch,
                    train_loss=position['loss_sum']/position['loss_count'] if position['loss_count'] else None,
                    sampled_labels=position['labels'], dev=result, successful=position['successful'],
                    attempts=position['attempts'], overflows=position['overflows']))
                emit('EPOCH_COMPLETE', model=name, epoch=epoch, **result)
                position.update(epoch=epoch + 1, offset=0, epoch_complete=False,
                                loss_sum=0., loss_count=0, labels={'0': 0, '1': 0})
            hashes = {p.name: digest(p) for p in (outdir/'best.pt', outdir/'last.pt', resume)}
            atomic_json(report/'COMPLETED.json', dict(epochs=t['epochs'], best_epoch=position['best_epoch'],
                best_dev=position['best'], sha256=hashes, session_seconds=time.monotonic()-started))
            atomic_json(report/'run_status.json', dict(status='completed', completed=True, position=position, test_accessed=False))
            emit('COMPLETED', model=name, best_epoch=position['best_epoch'], sha256=hashes)
        except BaseException as exc:
            # Do not overwrite a healthy recovery checkpoint with corrupted model state.
            text = traceback.format_exc()
            with (report/'errors.log').open('a') as stream:
                stream.write(text)
            print(text, file=sys.stderr, flush=True)
            atomic_json(report/'run_status.json', dict(status='stopped' if isinstance(exc, InterruptedError) else 'failed',
                error=str(exc), position=position, recovery=str(resume), completed=False, test_accessed=False))
            raise
    del model, optimizer, scheduler, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def dev_or(run_dir, checkpoint_dir):
    """Dev-only paired-eligible coverage; no media decoding or test prediction."""
    from src.datasets.frozen import FrozenMediaDataset
    quality = json.loads((ROOT/'configs/phase0_data_quality_v1.json').read_text())
    paired = FrozenMediaDataset('FAV', 'dev', eligibility_path=ROOT/quality['eligibility_manifest'],
                               registry=ROOT/'configs/fixed_manifests.json', task='paired')
    expected = {row['sample_id']: row for row in paired.rows if row['labels'][0] is not None}
    predictions = {}
    for name in ('x3d','aasist'):
        rows = [json.loads(line) for line in (run_dir/name/'best_dev.jsonl').read_text().splitlines()]
        validate_predictions(rows, [r['sample_id'] for r in rows])
        predictions[name] = {r['sample_id']: r for r in rows}
        if not expected.keys() <= predictions[name].keys():
            raise ValueError(f'Missing paired dev samples in {name}')
    rows=[]
    for sid, row in sorted(expected.items()):
        pv=predictions['x3d'][sid]['fake_probability']; pa=predictions['aasist'][sid]['fake_probability']
        rows.append(dict(sample_id=sid, split='dev', label=row['labels'][0],
                         video_probability=pv, audio_probability=pa, fake_probability=1-(1-pv)*(1-pa)))
    result=metrics([r['label'] for r in rows], [r['fake_probability'] for r in rows])
    directory=run_dir/'x3a_or';directory.mkdir(exist_ok=True)
    (directory/'dev_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    atomic_json(directory/'dev_summary.json',dict(metrics=result,threshold=.5,paired_samples=len(rows),
        checkpoints={n:digest(checkpoint_dir/n/'best.pt') for n in ('x3d','aasist')},test_accessed=False))
    emit('DEV_OR', **result)


class Parallel:
    """Small explicit collective boundary; object/error traffic uses CPU Gloo."""
    def __init__(self, control):
        import torch.distributed as dist
        self.dist = dist; self.control = control
        self.rank = dist.get_rank(); self.world = dist.get_world_size()

    def collect(self, value):
        values = [None] * self.world
        self.dist.all_gather_object(values, value, group=self.control)
        return values

    def check(self, error):
        errors = self.collect(error)
        if any(x is not None for x in errors):
            raise RuntimeError(f'Coordinated rank error: {errors}')

    def call(self, function):
        result = None; error = None
        try:
            result = function()
        except Exception:
            error = traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
        self.check(error)
        return result

    def root(self, function):
        result = self.call(lambda: function() if self.rank == 0 else None)
        return self.collect(result)[0]

    def buffers(self, model):
        for buffer in model.buffers():
            self.dist.broadcast(buffer, src=0)


def draw_shard(draws, rank, world, offset=0):
    if len(draws) % world or not 0 <= rank < world:
        raise ValueError('Training draws must divide evenly across devices')
    local = draws[rank::world]
    if not 0 <= offset <= len(local):
        raise ValueError('Invalid per-rank sampler offset')
    return local[offset:]


def save_ddp(path, model, optimizer, scheduler, scaler, position, provenance, device, parallel):
    parallel.buffers(model)
    states = parallel.collect(dict(rng=capture_rng(device), position=dict(position)))
    def write():
        payload = dict(format='x3a_ddp_v1', model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), provenance=provenance,
            rank_states=states, world_size=parallel.world)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        emit('CHECKPOINT', path=path, epoch=position['epoch'], offset=position['offset'], successful=position['successful'])
    parallel.root(write)


def validate_ddp_provenance(state, provenance, checkpoint_path, migration_file=None):
    """Only a registered, zero-update AASIST DDP fix may change source hash."""
    old = state['provenance']
    if old == provenance:
        return None
    import copy
    before, after = copy.deepcopy(old), copy.deepcopy(provenance)
    old_source = before.pop('unified_source_sha256', None)
    new_source = after.pop('unified_source_sha256', None)
    if before != after or old.get('config', {}).get('model', {}).get('name') != 'aasist':
        raise ValueError('DDP resume immutable provenance mismatch')
    positions = [r['position'] for r in state['rank_states']]
    if len(positions) != 4 or any(p.get(k) != 0 for p in positions
                                for k in ('epoch', 'offset', 'successful', 'attempts')):
        raise ValueError('Source migration requires the initial zero-update AASIST checkpoint')
    registry = Path(migration_file) if migration_file else ROOT/'configs/aasist_ddp_unused_fix.json'
    record = json.loads(registry.read_text()) if registry.is_file() else {}
    if (record.get('old_source_sha256') != old_source or
        record.get('new_source_sha256') != new_source or
        record.get('parent_checkpoint_sha256') != digest(checkpoint_path)):
        raise ValueError('Unregistered AASIST source/checkpoint migration')
    return record


def load_ddp(path, model, optimizer, scheduler, scaler, provenance, device, parallel):
    def read():
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('format') != 'x3a_ddp_v1' or state.get('world_size') != parallel.world:
            raise ValueError('Only matching unified 4-GPU checkpoints can resume')
        migration = validate_ddp_provenance(state, provenance, path)
        if migration:
            emit('SOURCE_MIGRATION', rank=parallel.rank, **migration)
        positions=[r['position'] for r in state['rank_states']]
        for key in ('epoch','offset','epoch_complete','successful','attempts'):
            if len({p[key] for p in positions}) != 1:
                raise ValueError('Checkpoint rank cursor mismatch')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        local=state['rank_states'][parallel.rank]
        restore_rng(local['rng'],device)
        return local['position']
    return parallel.call(read)


def train_distributed(name, args, run_dir, checkpoint_dir, device, parallel):
    from torch.nn.parallel import DistributedDataParallel
    config_path=ROOT/f'configs/experiments/x3a_reproduction/fav_indomain_v1/{name}_4gpu.yaml'
    task='video' if name=='x3d' else 'audio'
    report=run_dir/name; outdir=checkpoint_dir/name
    parallel.root(lambda:(report.mkdir(exist_ok=True),outdir.mkdir(exist_ok=True)))
    emit('PREPARE',rank=parallel.rank,model=name)
    config,train,dev,factory,collate,provenance=parallel.call(lambda:prepare(config_path,ROOT,task,device))
    t=config['training'];batch_size=t['batch_size'];accum=t['gradient_accumulation_steps']
    provenance['execution']=dict(mode='ddp',world_size=parallel.world,physical_devices=[0,1,2,3],
        per_device_batch=batch_size,accumulation=accum,effective_batch=batch_size*accum*parallel.world,
        learning_rate=t['optimizer']['learning_rate'])
    raw=parallel.call(lambda:factory().to(device))
    # Explicit buffer sync at dev/save boundaries; dev forward never uses DDP.
    model=DistributedDataParallel(raw,device_ids=[device.index] if device.type == 'cuda' else None,
        broadcast_buffers=False, find_unused_parameters=(name == 'aasist'))
    optimizer,scheduler,scaler=make_optimizer(raw,config,device)
    # Different dropout RNG streams, but identical initialized model parameters.
    torch.manual_seed(config['experiment']['seed']+parallel.rank)
    position=dict(epoch=0,offset=0,epoch_complete=False,successful=0,attempts=0,
        overflows=0,consecutive=0,best=None,best_epoch=None,loss_sum=0.,loss_count=0,labels={'0':0,'1':0})
    recovery=outdir/'recovery.pt'
    resume_stage=args.resume and not parallel.root(lambda:args.model=='all' and not any(report.iterdir()) and not any(outdir.iterdir()))
    if resume_stage:
        position=load_ddp(recovery,raw,optimizer,scheduler,scaler,provenance,device,parallel)
        emit('RESUME',rank=parallel.rank,**position)
    parallel.root(lambda:atomic_json(report/'run_manifest.json',provenance))
    last_print=[0.]
    def progress(stage,current,total,**fields):
        now=time.monotonic()
        if current not in (0,1,total) and now-last_print[0]<args.log_seconds:return
        last_print[0]=now
        status=dict(stage=stage,rank=parallel.rank,epoch=position['epoch'],batch=current,total=total,
            offset=position['offset'],successful=position['successful'],attempts=position['attempts'],
            scale=scaler.get_scale(),timestamp=time.time(),**fields)
        atomic_json(report/f'heartbeat_rank{parallel.rank}.json',status)
        emit(stage.upper(),**{k:v for k,v in status.items() if k!='stage'})
    def save(path):save_ddp(path,raw,optimizer,scheduler,scaler,position,provenance,device,parallel)
    with stop_requests() as stop:
        try:
            if not resume_stage:save(recovery)
            if position['epoch_complete']:
                position.update(epoch=position['epoch']+1,offset=0,epoch_complete=False,
                    loss_sum=0.,loss_count=0,labels={'0':0,'1':0})
            labels=[r['labels'][1 if task=='video' else 2] for r in train.rows]
            for epoch in range(position['epoch'],t['epochs']):
                position['epoch']=epoch
                draws=epoch_draws(labels,t['sampler']['global_epoch_samples'],config['experiment']['seed'],epoch)
                local=draw_shard(draws,parallel.rank,parallel.world,position['offset'])
                loader=DataLoader(Subset(train,local),batch_size=batch_size,collate_fn=collate,num_workers=0,
                    generator=torch.Generator().manual_seed(42+epoch))
                model.train();optimizer.zero_grad(set_to_none=True);pending=0
                progress('train',0,len(loader))
                iterator=iter(loader)
                for i in range(len(loader)):
                    batch=parallel.call(lambda:next(iterator))
                    sync=(i+1)%accum==0 or i+1==len(loader)
                    with nullcontext() if sync else model.no_sync():
                        def forward():
                            with torch.autocast(device.type,dtype=torch.float16,enabled=t['mixed_precision']):
                                out,y,ids=batch_forward(model,batch,device,task)
                                loss=(torch.nn.functional.binary_cross_entropy_with_logits(out.logits,y.float())
                                      if task=='video' else torch.nn.functional.cross_entropy(out.logits,y.long()))
                            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                            return loss,y
                        loss,y=parallel.call(forward)
                        group_samples=min(accum*batch_size,len(local)-(i//accum)*accum*batch_size)
                        scaler.scale(loss*(len(y)/group_samples)).backward()
                    pending+=len(y);position['loss_sum']+=float(loss.detach())*len(y);position['loss_count']+=len(y)
                    for label in y.tolist():position['labels'][str(label)]+=1
                    if sync:
                        record=optimizer_update(raw,optimizer,scaler,t,position['consecutive'],parallel)
                        decisions=parallel.collect((record['step_performed'],record['new_scale']))
                        parallel.check(None if len(set(decisions))==1 else 'Optimizer/AMP decision divergence')
                        position['attempts']+=1;position['successful']+=int(record['step_performed'])
                        position['overflows']+=int(record['overflow']);position['consecutive']=record['consecutive']
                        position['offset']+=pending;pending=0;optimizer.zero_grad(set_to_none=True)
                        def write_update():
                            with (report/f'updates_rank{parallel.rank}.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(record,epoch=epoch,offset=position['offset'],
                                    successful=position['successful'],attempt=position['attempts'],loss=float(loss.detach())))+'\n')
                        parallel.call(write_update)
                        if record['overflow'] or record['missing_gradient']:emit('GRADIENT',rank=parallel.rank,**record)
                        stopping=any(parallel.collect(stop['signal'] is not None))
                        if stopping or (record['step_performed'] and position['successful']%t.get('recovery_interval_steps',200)==0):save(recovery)
                        if stopping:raise InterruptedError('User requested coordinated stop; recovery saved')
                    parallel.call(lambda:progress('train',i+1,len(loader),loss=float(loss.detach())))
                save(recovery)
                # No per-batch collective: uneven dev shards safely finish independently.
                dev_subset=Subset(dev,list(range(parallel.rank,len(dev),parallel.world)))
                dev_loader=DataLoader(dev_subset,batch_size=batch_size,collate_fn=collate,num_workers=0,
                    generator=torch.Generator().manual_seed(0))
                raw.eval();records=[]
                def local_eval():
                    with torch.no_grad():
                        for i,batch in enumerate(dev_loader):
                            if stop['signal']:raise InterruptedError('Stop requested during dev; train-end recovery retained')
                            with torch.autocast(device.type,dtype=torch.float16,enabled=t['mixed_precision']):
                                output,y,ids=batch_forward(raw,batch,device,task)
                            records.extend(dict(sample_id=s,split='dev',label=l,logit=z,fake_probability=p,checkpoint_epoch=epoch)
                                for s,l,z,p in zip(ids,y.tolist(),output.logits.float().tolist(),output.fake_probability.float().tolist()))
                            progress('dev',i+1,len(dev_loader))
                parallel.call(local_eval)
                parts=parallel.collect(records);stats=parallel.collect(dict(position))
                def score():
                    rows=[r for part in parts for r in part]
                    validate_predictions(rows,[r['sample_id'] for r in dev.rows])
                    result=metrics([r['label'] for r in rows],[r['fake_probability'] for r in rows])
                    improved=better(result,position['best'])
                    pred=''.join(json.dumps(r)+'\n' for r in sorted(rows,key=lambda r:r['sample_id']))
                    (report/f'dev_{epoch:03d}.jsonl').write_text(pred)
                    if improved:(report/'best_dev.jsonl').write_text(pred)
                    count=sum(p['loss_count'] for p in stats)
                    atomic_json(report/f'epoch_{epoch:03d}.json',dict(epoch=epoch,dev=result,
                        train_loss=sum(p['loss_sum'] for p in stats)/count if count else None,
                        sampled_labels={key:sum(p['labels'][key] for p in stats) for key in ('0','1')},
                        rank_draws=[p['offset'] for p in stats],draw_position_overlap=0,
                        successful=position['successful'],attempts=position['attempts'],overflows=position['overflows']))
                    return result,improved
                result,improved=parallel.root(score)
                if improved:position['best']=result;position['best_epoch']=epoch
                scheduler.step();position['epoch_complete']=True
                save(outdir/'last.pt')
                if improved:save(outdir/'best.pt')
                save(recovery)
                if parallel.rank==0:emit('EPOCH_COMPLETE',epoch=epoch,**result)
                position.update(epoch=epoch+1,offset=0,epoch_complete=False,loss_sum=0.,loss_count=0,labels={'0':0,'1':0})
            def complete():
                hashes={p.name:digest(p) for p in (outdir/'last.pt',outdir/'best.pt',recovery)}
                atomic_json(report/'COMPLETED.json',dict(epochs=t['epochs'],best_epoch=position['best_epoch'],sha256=hashes))
                atomic_json(report/'run_status.json',dict(status='completed',completed=True,test_accessed=False))
                emit('COMPLETED',model=name,sha256=hashes)
            parallel.root(complete)
        except BaseException:
            text=traceback.format_exc();print(text,file=sys.stderr,flush=True)
            (report/f'error_rank{parallel.rank}.log').write_text(text)
            atomic_json(report/f'status_rank{parallel.rank}.json',dict(status='stopped_or_failed',position=position,completed=False))
            # Do not enter new NCCL collectives from a lone exception path.
            # Last healthy optimizer-boundary recovery remains untouched.
            raise
    del model,raw,optimizer,scheduler,scaler
    torch.cuda.empty_cache()


def distributed_main(args):
    import torch.distributed as dist
    from datetime import timedelta
    if os.environ.get('WORLD_SIZE')!='4' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3':
        raise ValueError('Use torchrun --nproc_per_node=4 with CUDA_VISIBLE_DEVICES=0,1,2,3')
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=120));parallel=Parallel(control)
    base=Path('experiments/x3a_reproduction/fav_indomain_v1/unified_4gpu')
    run_dir=ROOT/'reports'/base/args.run_id;checkpoint_dir=ROOT/'checkpoints'/base/args.run_id
    monitor=None
    try:
        def directories():
            if not args.resume and (run_dir.exists() or checkpoint_dir.exists()):raise FileExistsError('Use a new run-id or --resume')
            if args.resume and (not run_dir.is_dir() or not checkpoint_dir.is_dir()):raise FileNotFoundError('Resume run missing')
            run_dir.mkdir(parents=True,exist_ok=True);checkpoint_dir.mkdir(parents=True,exist_ok=True)
        parallel.root(directories)
        emit('START',rank=parallel.rank,physical_gpu=local,uuid=getattr(torch.cuda.get_device_properties(device),'uuid','see gpu_telemetry.jsonl'),report=run_dir)
        def start_monitor():
            nonlocal monitor
            from src.utils.gpu_monitor import GPUMonitor,ThermalPolicy
            monitor=GPUMonitor(run_dir,'gpu_telemetry.jsonl',interval_seconds=10,policy=ThermalPolicy(80,88,1,10),stop_enabled=False)
            monitor.start()
        parallel.root(start_monitor)
        for name in (('x3d','aasist') if args.model=='all' else (args.model,)):
            train_distributed(name,args,run_dir,checkpoint_dir,device,parallel)
        if args.model=='all':parallel.root(lambda:dev_or(run_dir,checkpoint_dir))
    finally:
        if monitor:monitor.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', type=int, choices=[1,4], default=1)
    parser.add_argument('--model', choices=['x3d', 'aasist', 'all'], default='x3d')
    parser.add_argument('--run-id', required=True, help='New run directory name; use the same ID with --resume')
    parser.add_argument('--resume', action='store_true', help='Resume a matching unified single-GPU or 4-GPU checkpoint; legacy checkpoints rejected')
    parser.add_argument('--log-seconds', type=float, default=10.)
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in ('.','..') or args.log_seconds <= 0:
        parser.error('Invalid run ID/log interval')
    if args.devices == 4:
        return distributed_main(args)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or len(visible.split(',')) != 1:
        parser.error('Set CUDA_VISIBLE_DEVICES to exactly one physical GPU, for example 1')
    base = Path('experiments/x3a_reproduction/fav_indomain_v1/unified_single_gpu')
    run_dir = ROOT/'reports'/base/args.run_id
    checkpoint_dir = ROOT/'checkpoints'/base/args.run_id
    if not args.resume and (run_dir.exists() or checkpoint_dir.exists()):
        parser.error('Output exists: choose a new run ID or explicitly --resume')
    if args.resume and (not run_dir.is_dir() or not checkpoint_dir.is_dir()):
        parser.error('Resume directory does not exist')
    if not torch.cuda.is_available():
        parser.error('CUDA unavailable; run in the host terminal')
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    run_dir.mkdir(parents=True,exist_ok=True);checkpoint_dir.mkdir(parents=True,exist_ok=True)
    emit('START', model=args.model, physical_gpu=visible, report=run_dir, checkpoints=checkpoint_dir)
    from src.utils.gpu_monitor import GPUMonitor, ThermalPolicy
    monitor = GPUMonitor(run_dir, 'gpu_telemetry.jsonl', interval_seconds=10,
                         policy=ThermalPolicy(80,88,1,10), stop_enabled=False)
    monitor.start()
    try:
        for name in (('x3d','aasist') if args.model == 'all' else (args.model,)):
            train_model(name, args, run_dir, checkpoint_dir, device)
        if args.model == 'all':
            dev_or(run_dir, checkpoint_dir)
    finally:
        monitor.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
