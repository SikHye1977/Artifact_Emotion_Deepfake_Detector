"""FAV only: reuse artifact caches -> emotion variation -> train/dev -> locked test."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.hierarchical_score_fusion_variation.emotion2vec_seed_script import load_artifact_cache,no_overlap,write_json,sha256,code_hashes,Logger
from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import download_seed
from src.models.hierarchical_score_fusion_variation.artifact_provenance import artifact_recipe_digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True,help='Contains cache_train/cache_dev/cache_test from the previous artifact pipeline')
    p.add_argument('--eligibility',type=Path,required=True)
    p.add_argument('--registry',type=Path,default=ROOT/'configs/fixed_manifests.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',default=['0','1','2','3'])
    p.add_argument('--head-device',default='cpu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--revision',default='main')
    p.add_argument('--seed-evaluation',type=Path,help='Prior seed run_manifest.json; verify and reuse its exact local snapshot offline')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output; prior results are preserved')
    if args.epochs<1 or not args.gpus:raise ValueError('Invalid epochs/GPUs')
    # Validate all artifact inputs before downloading/loading emotion models.
    caches={};hashes={}
    for split in ('train','dev','test'):
        path=args.artifact_root/f'cache_{split}'
        data,meta=load_artifact_cache(path,split)
        if meta.get('dataset')!='FAV':raise ValueError('Only FAV is permitted')
        if sha256(args.eligibility)!=meta['recipe']['eligibility_sha256'] or sha256(args.registry)!=meta['recipe']['registry_sha256']:
            raise ValueError('Eligibility/registry differs from artifact run')
        caches[split]=(data,meta);hashes[split]=meta['cache_sha256']
    if len({artifact_recipe_digest(m['recipe']) for _,m in caches.values()})!=1:raise ValueError('Artifact split recipes differ')
    for a,b in [('train','dev'),('train','test'),('dev','test')]:no_overlap(caches[a][0],caches[b][0])
    args.output.mkdir(parents=True);log=Logger(args.output)
    try:
        write_json(args.output/'artifact_recipe_audit.json',dict(policy='reviewed_source_relocation_only_v1',splits={split:dict(original_recipe_sha256=m['recipe_sha256'],canonical_recipe_sha256=artifact_recipe_digest(m['recipe'])) for split,(_,m) in caches.items()}))
        log('reuse_seed_offline' if args.seed_evaluation else 'download_seed')
        snapshot,commit=download_seed(args.revision,args.seed_evaluation)
        config=dict(audio_model='emotion2vec_plus_seed',seed_snapshot=snapshot,revision=commit,
                    video_classes=8,audio_classes=9,window_seconds=3,hop_seconds=1)
        write_json(args.output/'resolved_config.json',config)
        write_json(args.output/'plan.json',dict(dataset='FAV',seed=args.seed,epochs=args.epochs,gpus=args.gpus,
            implementation=code_hashes(),artifact_cache_hashes=hashes,seed_revision=commit,
            policy='frozen encoders; per-modality train labels; dev-AUC head selection; threshold0.5 OR; no test tuning'))
        script=Path(__file__).with_name('emotion2vec_seed_script.py')
        def extract(split):
            log('pipeline_stage',stage='extract_'+split)
            subprocess.run([sys.executable,str(script),'extract-multi','--config',str((args.output/'resolved_config.json').resolve()),
                '--artifact-cache',str((args.artifact_root/f'cache_{split}').resolve()),'--eligibility',str(args.eligibility.resolve()),
                '--registry',str(args.registry.resolve()),'--split',split,'--gpus',*args.gpus,
                '--output',str((args.output/f'variation_{split}').resolve())],check=True)
        for split in ('train','dev'):extract(split)
        log('pipeline_stage',stage='train_and_dev')
        subprocess.run([sys.executable,str(script),'train','--train-cache',str((args.output/'variation_train').resolve()),
            '--dev-cache',str((args.output/'variation_dev').resolve()),'--seed',str(args.seed),'--epochs',str(args.epochs),
            '--device',args.head_device,'--output',str((args.output/'heads').resolve())],check=True)
        checkpoint=args.output/'heads/emotion_heads.pt'
        write_json(args.output/'LOCKED_BEFORE_TEST.json',dict(checkpoint_sha256=sha256(checkpoint),threshold=.5,
            final_fusion='probabilistic_OR',implementation=code_hashes()))
        extract('test')
        log('pipeline_stage',stage='test')
        subprocess.run([sys.executable,str(script),'evaluate','--cache',str((args.output/'variation_test').resolve()),
            '--checkpoint',str(checkpoint.resolve()),'--split','test','--device',args.head_device,
            '--output',str((args.output/'test').resolve())],check=True)
        write_json(args.output/'COMPLETED.json',dict(dataset='FAV',test_access=True,checkpoint_sha256=sha256(checkpoint)))
        log('pipeline_complete',results=str(args.output/'test/test_metrics.json'))
    except BaseException as error:
        import traceback
        (args.output/'error.txt').write_text(traceback.format_exc())
        write_json(args.output/'FAILED.json',dict(error=repr(error)))
        raise
if __name__=='__main__':main()
