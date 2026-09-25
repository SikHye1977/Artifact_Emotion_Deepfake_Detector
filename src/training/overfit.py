"""Train-only fixed-subset pipeline verification, never an experiment result."""
import gc
import json
import random
import tempfile
import time
from pathlib import Path

import torch
from .trainer import Trainer, train_epoch
from .losses import modality_loss


def select_subset(rows, label_index, seed=42):
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('Overfit accepts train rows only')
    ordered = sorted(rows, key=lambda row: row['sample_id'])
    if len({row['sample_id'] for row in ordered}) != len(ordered):
        raise ValueError('Duplicate sample_id')
    rng = random.Random(seed)
    groups = []
    for label in (0, 1):
        pool = [row for row in ordered if row['labels'][label_index] == label]
        if len(pool) < 8:raise ValueError('Need eight eligible train samples of each label')
        groups.append(rng.sample(pool, 8))
    # Fixed mixed-class batches; no replacement or weighted sampler.
    return [row for pair in zip(*groups) for row in pair]


@torch.no_grad()
def predict_subset(model, batches, step, task, device, amp):
    model.eval(); records=[]; total=0.0
    for batch in batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output, labels, ids = step(model,batch,device)
            loss = modality_loss(output.logits, labels, task)
        if not torch.isfinite(loss) or not torch.isfinite(output.logits).all():
            raise FloatingPointError('Nonfinite subset evaluation')
        total += float(loss) * len(ids)
        for sid, label, logits, prob in zip(ids,labels.cpu().tolist(),output.logits.float().cpu().tolist(),output.fake_probability.float().cpu().tolist()):
            records.append(dict(sample_id=sid,label=label,logits=logits,fake_probability=prob))
    accuracy=sum((r['fake_probability'] >= .5) == r['label'] for r in records)/len(records)
    return dict(loss=total/len(records),accuracy=accuracy,predictions=records)


def run_subset_overfit(dataset, factory, config, step, task, device, provenance, collate, root):
    name='x3d' if task=='video' else 'aasist';index=1 if task=='video' else 2
    report=Path(root)/'reports/phase1_preflight/small_subset_overfit.json'
    log=report.parent/(name+'_small_subset_loss.jsonl')
    if log.exists():raise FileExistsError(f'Preserve previous overfit log: {log}')
    dataset.rows=select_subset(dataset.rows,index)
    t=config['training'];physical=t['batch_size'];accum=t['gradient_accumulation_steps']
    torch.manual_seed(42);random.seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    result=dict(status='running',scope='train-only learning pipeline check, not model performance experiment',
        seed=42,selection='sort sample_id; random.Random(42).sample 8 per class, real then fake; interleave selected classes',
        subset=[{'sample_id':r['sample_id'],'label':r['labels'][index]} for r in dataset.rows],
        label_key=task+'_label',physical_batch=physical,accumulation_steps=accum,
        max_optimizer_steps=200,stopping_rule='subset eval accuracy >= 0.95 AND eval loss <= 0.2 * initial eval loss',
        random_augmentation=False,dropout='model-native train dropout retained, RNG seed fixed',
        scheduler='constant configured learning rate for this bounded diagnostic; epoch scheduler not advanced',
        preprocessing='fixed YAML recipe, cache only selected decoded train tensors in RAM',
        provenance=provenance,log_path=str(log.relative_to(root)))
    def save():
        previous=json.loads(report.read_text()) if report.exists() else {}
        previous[name]=result;report.write_text(json.dumps(previous,indent=2,allow_nan=False)+'\n')
    save();start=time.monotonic();trainer=None;last=None;completed=0
    try:
        # The only dataset __getitem__ calls in this check: selected train IDs.
        items=[dataset[i] for i in range(16)]
        batches=[collate(items[i:i+physical]) for i in range(0,16,physical)]
        del items
        trainer=Trainer(factory().to(device),config,step,task,device,provenance,torch.Generator().manual_seed(42))
        if device.type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        initial=predict_subset(trainer.model,batches,step,task,device,t['mixed_precision'])
        result['initial_loss']=initial['loss'];minimum=initial['loss'];last=initial
        with log.open('x') as output:
            output.write(json.dumps(dict(optimizer_step=0,subset=initial))+'\n');output.flush()
            for update in range(1,201):
                micro=[batches[((update-1)*accum+j)%len(batches)] for j in range(accum)]
                stats=train_epoch(trainer.model,micro,step,task,trainer.optimizer,trainer.scaler,device,
                                  accumulation_steps=accum,amp=t['mixed_precision'])
                completed=update;trainer.global_step=update
                last=predict_subset(trainer.model,batches,step,task,device,t['mixed_precision'])
                minimum=min(minimum,last['loss'])
                output.write(json.dumps(dict(optimizer_step=update,training=stats,subset=last))+'\n');output.flush()
                result.update(optimizer_steps=update,final_loss=last['loss'],minimum_loss=minimum,accuracy=last['accuracy'])
                save()
                if update%5==0:print(name,update,'subset loss',last['loss'],'accuracy',last['accuracy'],flush=True)
                if last['accuracy']>=.95 and last['loss']<=initial['loss']*.2:
                    result['status']='overfit_pass';break
            else:result['status']='max_steps_reached_without_overfit_criterion'
        # Keep only the last state; no dev metric or best/last experiment files.
        if device.type=='cuda':
            result['cuda_peak_allocated']=torch.cuda.max_memory_allocated()
            result['cuda_peak_reserved']=torch.cuda.max_memory_reserved()
        with tempfile.TemporaryDirectory(prefix='subset-overfit-checkpoint-') as directory:
            cp=Path(directory)/'overfit_state.pt';trainer.save(cp)
            del trainer;trainer=None;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            restored=Trainer(factory().to(device),config,step,task,device,provenance,torch.Generator())
            restored.resume(cp)
            actual=predict_subset(restored.model,batches,step,task,device,t['mixed_precision'])
            assert [r['sample_id'] for r in last['predictions']]==[r['sample_id'] for r in actual['predictions']]
            before=torch.tensor([r['logits'] for r in last['predictions']]);after=torch.tensor([r['logits'] for r in actual['predictions']])
            torch.testing.assert_close(before,after,rtol=0,atol=0)
            result['checkpoint_verification']={'exact_logit_match':True,'max_abs_difference':float((before-after).abs().max()),'restored_global_step':restored.global_step,'temporary_checkpoint_deleted':True}
            del restored
    except (torch.cuda.OutOfMemoryError,FloatingPointError,RuntimeError,AssertionError) as exc:
        result.update(status='oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'failed',error=repr(exc),optimizer_steps=completed)
        if device.type=='cuda':
            result['cuda_peak_allocated']=torch.cuda.max_memory_allocated();result['cuda_peak_reserved']=torch.cuda.max_memory_reserved()
    finally:
        result['elapsed_seconds']=time.monotonic()-start;save()
    print(name,result['status'],flush=True)
    return result
