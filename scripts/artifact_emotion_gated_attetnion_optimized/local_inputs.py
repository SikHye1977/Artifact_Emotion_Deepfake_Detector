"""Inputs for a new local FAV experiment; no source-workstation hash dependency."""
from pathlib import Path
import json
import torch
from .common import ROOT, read
from src.utils.local_paths import resolve_media_path


def local_rows(registry, eligibility, modality, split=None):
    entry = read(registry)['datasets']['FAV']
    quality = {}
    with Path(eligibility).open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['dataset'] == 'FAV':
                if row['sample_id'] in quality:raise ValueError('Duplicate eligibility ID')
                quality[row['sample_id']] = row
    rows=[]; seen=set(); group_splits={}
    with (ROOT / entry['manifest']).open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line); sid=row['sample_id']; part=row['split']
            if sid in seen:raise ValueError('Duplicate manifest ID')
            seen.add(sid)
            if part not in ('train','dev','test'):raise ValueError('Invalid split')
            group=row.get('source_id')
            if group is None or str(group)=='':raise ValueError('Missing source_id')
            group=str(group)
            if group in group_splits and group_splits[group]!=part:raise ValueError('Group leakage across splits')
            group_splits[group]=part
            labels=[row[k] for k in ('clip_label','video_label','audio_label')]
            if any(v not in (None,-1,0,1) for v in labels):raise ValueError('Invalid label')
            q=quality.get(sid)
            if q is None:raise ValueError('Missing eligibility record: '+sid)
            if not q['eligible'][modality] or (split is not None and part!=split):continue
            rows.append(dict(sample_id=sid,group=group,path=str(resolve_media_path(row['video_path'])),
                             split=part,labels=labels))
    if not rows:raise ValueError('No eligible local rows')
    return rows


def verify_local_file(row):
    path=Path(row['path'])
    if not path.is_file() or path.stat().st_size==0:
        raise FileNotFoundError('Missing/empty local media: '+str(path))


def load_local_artifact(checkpoint, modality):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state.get('format') not in ('x3a_ddp_v1','x3a_single_v1'):raise ValueError('Unexpected artifact checkpoint format')
    cfg=state['provenance']['config']
    if modality=='video':
        from src.models.x3d import X3DDeepfakeClassifier
        if cfg['model']['name']!='x3d_m' or cfg['video']['num_frames']!=128:raise ValueError('Unexpected video architecture')
        model=X3DDeepfakeClassifier('x3d_m',pretrained=False)
    else:
        from src.models.aasist import build_aasist
        m=cfg['model']
        if m['fake_class_index']!=1 or m['class_mapping']!={'real':0,'fake':1}:raise ValueError('Unexpected audio label mapping')
        model=build_aasist(ROOT/cfg['third_party']['path'],m['official_config'],fake_class_index=1)
    weights=state['model']
    if weights and all(k.startswith('module.') for k in weights):weights={k[7:]:v for k,v in weights.items()}
    model.load_state_dict(weights,strict=True)
    return model.eval().requires_grad_(False),cfg
