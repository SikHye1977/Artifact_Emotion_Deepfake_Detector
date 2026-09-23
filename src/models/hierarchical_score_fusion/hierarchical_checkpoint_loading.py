"""Strict adapter for the previously shared reproduce_x3a checkpoint format."""
import hashlib
import json
from pathlib import Path
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_artifact(path, modality, root, trusted=False):
    if not trusted:
        raise ValueError('Existing training checkpoints may contain pickle: use --trusted-artifact-checkpoints for your own files')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format') not in ('x3a_single_v1', 'x3a_ddp_v1'):
        raise ValueError('Unknown reproduce_x3a format. Adapt this loader to the actual checkpoint; do not guess keys.')
    prov = state['provenance']
    cfg = prov['config']
    registry = root / cfg['data']['split_manifest']
    quality = json.loads((root / cfg['data']['eligibility']).read_text())
    eligibility = root / quality['eligibility_manifest']
    for file, expected in [(registry, prov['registry_sha256']), (eligibility, prov['eligibility_sha256'])]:
        if sha256(file) != expected:
            raise ValueError(f'Training data provenance changed: {file}')
    entry = json.loads(registry.read_text())['datasets']['FAV']
    if sha256(root / entry['manifest']) != entry['sha256'] or entry['sha256'] != prov['split_manifest_sha256']:
        raise ValueError('FAV manifest changed')
    for relative, expected in prov.get('model_source_sha256', {}).items():
        if relative in (f'src/models/{"x3d" if modality == "video" else "aasist"}.py',
                        'src/datasets/media.py', 'src/datasets/frozen.py'):
            if sha256(root / relative) != expected:
                raise ValueError(f'Training implementation changed: {relative}')
    if modality == 'video':
        from src.models.x3d import X3DDeepfakeClassifier
        if cfg['model']['name'] != 'x3d_m' or cfg['video']['num_frames'] != 128:
            raise ValueError('Expected the trained 128-frame X3D-M recipe')
        model = X3DDeepfakeClassifier('x3d_m', pretrained=False)
    else:
        from src.models.aasist import build_aasist
        m = cfg['model']
        if m['fake_class_index'] != 1 or m['class_mapping'] != {'real': 0, 'fake': 1}:
            raise ValueError('Expected trained AASIST real=0 fake=1')
        model = build_aasist(root / cfg['third_party']['path'], m['official_config'], fake_class_index=1)
    weights = state['model']
    if weights and all(k.startswith('module.') for k in weights):
        weights = {k[7:]: v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False).eval()
    return model, cfg, registry, eligibility
