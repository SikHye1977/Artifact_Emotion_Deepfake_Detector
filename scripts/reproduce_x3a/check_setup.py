"""CPU prerequisite checks only: no training, test inference or split regeneration."""
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    import yaml
    from scripts.reproduce_x3a.train_x3a import digest
    from scripts.reproduce_x3a import train_x3a, evaluate_x3a_final
    from src.datasets.media import MediaConfig
    from pytorchvideo.models.hub import x3d_m
    assert train_x3a.ROOT == evaluate_x3a_final.ROOT == ROOT
    local=yaml.safe_load((ROOT/'configs/local_paths.yaml').read_text())
    assert Path(local['fav_root']).is_dir(), 'Configure fav_root'
    quality=json.loads((ROOT/'configs/phase0_data_quality_v1.json').read_text())
    registry=ROOT/'configs/fixed_manifests.json'
    fav=json.loads(registry.read_text())['datasets']['FAV']
    for path,expected in [(registry,quality['fixed_manifest_registry_sha256']),
        (ROOT/quality['eligibility_manifest'],quality['eligibility_manifest_sha256']),
        (ROOT/fav['manifest'],fav['sha256'])]:
        assert digest(path)==expected, f'Hash mismatch: {path}'
        print('OK',path.relative_to(ROOT),flush=True)
    recipe=json.loads((ROOT/'reports/phase1_preflight/input_recipe.json').read_text())
    assert recipe['audio']['target_num_samples']==284672
    lock=json.loads((ROOT/'configs/third_party.lock.json').read_text())['aasist']
    vendor=ROOT/lock['path']
    assert subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()==lock['commit']
    assert not subprocess.check_output(['git','-C',str(vendor),'status','--porcelain'],text=True).strip()
    for binary in ('ffmpeg','ffprobe'):
        subprocess.run([binary,'-version'],check=True,stdout=subprocess.DEVNULL)
    print('SETUP_OK: CPU checks only; full media and CUDA not tested.')

if __name__=='__main__':main()
