"""Shared strict training loop; only train and dev loaders are accepted."""
from pathlib import Path
import json
import torch
import time
from collections import Counter
from src.evaluation.evaluate import evaluate
from src.utils.checkpointing import save_checkpoint,load_checkpoint
from .losses import modality_loss


def make_optimization(model, config, device):
    c=config['training'];o=c['optimizer']
    cls={'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[o['name']]
    opt=cls([p for p in model.parameters() if p.requires_grad],lr=o['learning_rate'],weight_decay=o['weight_decay'])
    if c['scheduler']['name'] != 'cosine':raise ValueError('Unsupported scheduler')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['epochs'])
    scaler=torch.amp.GradScaler(device.type,enabled=c['mixed_precision'],
        init_scale=float(c.get('amp_initial_scale',65536.0)))
    return opt,scheduler,scaler


def train_epoch(model,loader,step,task,optimizer,scaler,device,*,accumulation_steps,amp):
    if accumulation_steps < 1:raise ValueError('Invalid accumulation')
    model.train();optimizer.zero_grad(set_to_none=True);steps=0;losses=[]
    from .amp import GradientPolicy
    policy=GradientPolicy(); pending_ids=[]
    started=time.monotonic();labels=Counter()
    for i,batch in enumerate(loader):
        with torch.autocast(device.type,enabled=amp,dtype=torch.float16):
            out,y,ids=step(model,batch,device)
            labels.update(y.detach().cpu().tolist())
            loss=modality_loss(out.logits,y,task)
        if not torch.isfinite(loss) or not torch.isfinite(out.logits).all():raise FloatingPointError('Nonfinite loss/output')
        pending_ids.extend(ids)
        scaler.scale(loss/accumulation_steps).backward()
        if (i+1)%accumulation_steps==0 or i+1==len(loader):
            record=policy.finish(model,optimizer,scaler,pending_ids)
            optimizer.zero_grad(set_to_none=True);steps+=int(record['step_performed']);pending_ids=[]
        losses.append(float(loss.detach()))
        if (i+1)%100==0:print('train batch',i+1,'/',len(loader),'mean_loss',sum(losses)/len(losses),'elapsed',time.monotonic()-started,flush=True)
    elapsed=time.monotonic()-started
    return dict(optimizer_steps=steps,mean_loss=sum(losses)/len(losses),finite_gradients=True,
                sampled_label_counts=dict(labels),train_seconds=elapsed,mean_batch_seconds=elapsed/len(losses),amp_steps=policy.history)


class Trainer:
    def __init__(self,model,config,step,task,device,provenance,generator):
        self.model,self.config,self.step,self.task,self.device=model,config,step,task,device
        self.provenance,self.generator=provenance,generator
        self.optimizer,self.scheduler,self.scaler=make_optimization(model,config,device)
        self.epoch=-1;self.global_step=0;self.best=float('-inf')

    def resume(self,path):
        s=load_checkpoint(path,self.model,self.optimizer,self.scheduler,self.scaler,provenance=self.provenance,generator=self.generator)
        self.epoch,self.global_step,self.best=s['epoch'],s['global_step'],s['best_dev_metric']

    def save(self,path):
        save_checkpoint(path,self.model,self.optimizer,self.scheduler,self.scaler,epoch=self.epoch,
            global_step=self.global_step,best_dev_metric=self.best,provenance=self.provenance,generator=self.generator)

    def checkpoint_epoch(self,metric,directory):
        if metric>self.best:
            self.best=metric;self.save(Path(directory)/'best.pt')
        self.save(Path(directory)/'last.pt')

    def fit(self,train_loader,dev_loader,checkpoint_dir,report_dir,max_epochs=None):
        if {r['split'] for r in train_loader.dataset.rows}!={'train'} or {r['split'] for r in dev_loader.dataset.rows}!={'dev'}:
            raise ValueError('Trainer requires train/dev only')
        report_dir=Path(report_dir);report_dir.mkdir(parents=True,exist_ok=True)
        stop=self.config['training']['epochs'] if max_epochs is None else min(self.config['training']['epochs'],self.epoch+1+max_epochs)
        for epoch in range(self.epoch+1,stop):
            epoch_started=time.monotonic()
            stats=train_epoch(self.model,train_loader,self.step,self.task,self.optimizer,self.scaler,self.device,
                accumulation_steps=self.config['training']['gradient_accumulation_steps'],amp=self.config['training']['mixed_precision'])
            self.global_step+=stats['optimizer_steps'];self.epoch=epoch
            dev_started=time.monotonic()
            predictions,metrics=evaluate(self.model,dev_loader,self.step,self.device,self.config['training']['mixed_precision'])
            dev_seconds=time.monotonic()-dev_started
            self.scheduler.step();self.checkpoint_epoch(metrics['roc_auc'],checkpoint_dir)
            elapsed=time.monotonic()-epoch_started
            (report_dir/f'epoch_{epoch:03d}.json').write_text(json.dumps(dict(train=stats,dev=metrics,
                dev_seconds=dev_seconds,epoch_seconds=elapsed,estimated_20_epoch_seconds=elapsed*20,
                train_eligible_samples=len(train_loader.dataset),dev_samples=len(predictions),
                prediction_id_coverage='passed',execution_max_epochs=max_epochs),indent=2))
            with (report_dir/f'dev_{epoch:03d}.jsonl').open('w') as f:
                for row in predictions:f.write(json.dumps(row)+'\n')


def run_training(config_path,project_root,task,step,*,dry_run=False,resume_checkpoint=None,overfit=False,max_epochs=None,prepare_only=False):
    """Prepare immutable data and factories, then dry-run or explicitly train."""
    import gc, tempfile, hashlib, inspect, subprocess
    from functools import partial
    import yaml
    from torch.utils.data import DataLoader,WeightedRandomSampler
    from src.utils.config import load_experiment,sha256,resolve_sampler_policy,sampler_policy_sha256,validate_sampler_policy
    from src.utils.checkpointing import checkpoint_metadata
    from src.utils.reproducibility import seed_everything
    from src.datasets.frozen import FrozenMediaDataset
    from src.datasets.media import MediaConfig,collate_media
    from src.datasets.transforms import balanced_weights
    if max_epochs is not None and (type(max_epochs) is not int or max_epochs<1):raise ValueError('max_epochs must be positive')
    root=Path(project_root).resolve();config,_=load_experiment(config_path,root)
    name='x3d' if task=='video' else 'aasist';t=config['training']
    required_batch=(2, t['gradient_accumulation_steps'], True) if task=='video' else (4, t['gradient_accumulation_steps'], False)
    if (t['batch_size'],t['mixed_precision'])!=(t['per_device_batch_size'],required_batch[2]):
        raise ValueError('Training does not match input_recipe batch/precision')
    sampler_policy=resolve_sampler_policy(config)
    validate_sampler_policy(sampler_policy,task=task,expected_world_size=config.get('distributed',{}).get('world_size',4))
    if config['checkpoint_selection']!={'split':'dev','metric':'roc_auc','mode':'max'}:
        raise ValueError('Expected best dev ROC-AUC selection')
    if dry_run and resume_checkpoint:raise ValueError('Dry-run and resume are separate modes')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if t['mixed_precision'] and device.type!='cuda':raise RuntimeError('CUDA required for configured AMP')
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
    generator=torch.Generator().manual_seed(config['experiment']['seed'])
    label_index=1 if task=='video' else 2
    def dataset(split):
        ds=FrozenMediaDataset('FAV',split,media,eligibility_path=eligibility,task=task,registry=registry)
        ds.rows=[row for row in ds.rows if row['labels'][label_index] is not None]
        # Relocate by dataset-relative category path without modifying the manifest.
        for row in ds.rows:
            relative=row['sample_id'].split(':',1)[1]
            row['path']=str(Path(local['fav_root'])/relative)
        return ds
    train=dataset('train')
    if prepare_only:
        return dict(config=config,train=train,dataset=dataset,factory=factory,collate=collate,provenance=provenance)
    if 'distributed' in config and not dry_run and not overfit:
        raise RuntimeError('Four-GPU recipe requires the DDP entry point; single-process formal execution is disabled')
    if overfit:
        if dry_run or resume_checkpoint:raise ValueError('Overfit is a separate pipeline check')
        from .overfit import run_subset_overfit
        return run_subset_overfit(train, factory, config, step, task, device, provenance, collate, root)
    weights=balanced_weights([{task+'_label':r['labels'][label_index]} for r in train.rows],task+'_label')
    sampler=WeightedRandomSampler(weights,len(train),replacement=True,generator=generator)
    train_loader=DataLoader(train,batch_size=t['batch_size'],sampler=sampler,collate_fn=collate,num_workers=0)
    model=factory()
    initial_digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        initial_digest.update(key.encode());initial_digest.update(str(value.dtype).encode())
        initial_digest.update(str(tuple(value.shape)).encode());initial_digest.update(value.cpu().numpy().tobytes())
    provenance['initial_model_state_sha256']=initial_digest.hexdigest()
    model=model.to(device);trainer=Trainer(model,config,step,task,device,provenance,generator)
    if dry_run:
        batch=next(iter(train_loader));model.zero_grad(set_to_none=True)
        if device.type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        stats=train_epoch(model,[batch],step,task,trainer.optimizer,trainer.scaler,device,
                          accumulation_steps=t['gradient_accumulation_steps'],amp=t['mixed_precision'])
        trainer.global_step=stats['optimizer_steps']
        model.eval()
        with torch.no_grad():reference=step(model,batch,device)[0].logits.detach().cpu()
        peaks=dict(peak_allocated=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                   peak_reserved=torch.cuda.max_memory_reserved() if device.type=='cuda' else None)
        with tempfile.TemporaryDirectory(prefix='trainer-dry-run-') as temporary:
            path=Path(temporary)/'roundtrip.pt';trainer.save(path)
            # Release GPU optimizer/model before restoring a genuinely new instance.
            model.cpu();del trainer,model;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            restored=factory().to(device);trainer=Trainer(restored,config,step,task,device,provenance,generator);trainer.resume(path)
            restored.eval()
            with torch.no_grad():actual=step(restored,batch,device)[0].logits.detach().cpu()
            torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-6)
            result=dict(status='pass',sample_ids=batch['sample_id'],stats=stats,output_shape=list(actual.shape),
                checkpoint_restored=True,max_abs_difference=float((actual-reference).abs().max()),cuda=peaks,
                provenance=provenance,device=str(device),scope='one real train batch only; no dev/test evaluation')
        report=root/'reports/phase1_preflight/trainer_dry_run.json';report.parent.mkdir(parents=True,exist_ok=True)
        previous=json.loads(report.read_text()) if report.exists() else {};previous[name]=result
        report.write_text(json.dumps(previous,indent=2)+'\n')
        (report.parent/(name+'_dry_run_manifest.json')).write_text(json.dumps(provenance,indent=2)+'\n')
        return result
    if resume_checkpoint:trainer.resume(resume_checkpoint)
    elif any((root/config['outputs']['checkpoint_dir']/file).exists() for file in ('best.pt','last.pt')):
        raise FileExistsError('Existing run: explicitly resume rather than overwrite')
    dev_loader=DataLoader(dataset('dev'),batch_size=t['batch_size'],shuffle=False,collate_fn=collate,num_workers=0)
    report=root/config['outputs']['report_dir'];report.mkdir(parents=True,exist_ok=True)
    (report/'run_manifest.json').write_text(json.dumps(provenance,indent=2))
    trainer.fit(train_loader,dev_loader,root/config['outputs']['checkpoint_dir'],report,max_epochs=max_epochs)
