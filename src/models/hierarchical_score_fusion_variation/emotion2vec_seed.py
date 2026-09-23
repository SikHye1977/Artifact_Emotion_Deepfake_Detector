"""Frozen emotion2vec+ seed probabilities; canonical nine-class ordering."""
from pathlib import Path
import numpy as np
from src.models.hierarchical_score_fusion.hierarchical_checkpoint_loading import sha256

CLASSES=['angry','disgusted','fearful','happy','neutral','other','sad','surprised','unknown']
REPO='emotion2vec/emotion2vec_plus_seed'

def ordered_probabilities(result):
    names=[str(x).split('/')[-1].strip().lower() for x in result['labels']]
    names=[{'<unk>':'unknown','disgust':'disgusted','fear':'fearful','surprise':'surprised'}.get(x,x) for x in names]
    p=np.asarray(result['scores'],dtype=np.float32).reshape(-1)
    if len(names)!=9 or len(set(names))!=9 or set(names)!=set(CLASSES) or p.shape!=(9,):
        raise ValueError(f'Unexpected emotion2vec labels: {names}')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isclose(p.sum(),1,atol=1e-5):
        raise ValueError('Invalid emotion2vec probabilities')
    return p[[names.index(c) for c in CLASSES]]

def download_seed(revision='main', evaluation_manifest=None):
    from huggingface_hub import HfApi,snapshot_download
    if evaluation_manifest is not None:
        import json
        metadata=json.loads(Path(evaluation_manifest).read_text())['model']
        if metadata['repo']!=REPO:raise ValueError('Evaluation manifest is not emotion2vec+ seed')
        commit=metadata['revision']
        if not isinstance(commit,str) or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
            raise ValueError('Expected immutable snapshot commit')
        path=snapshot_download(REPO,revision=commit,local_files_only=True,token=False)
        files=metadata['files_sha256']
        if 'model.pt' not in files or 'config.yaml' not in files:raise ValueError('Incomplete evaluation weight provenance')
        root=Path(path)
        for name,expected in files.items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid snapshot path')
            if sha256(root/relative)!=expected:raise ValueError(f'Cached evaluation file hash mismatch: {name}')
        return path,commit
    # This public repository does not require a token; do not send an expired cached credential.
    commit=HfApi().model_info(REPO,revision=revision,token=False).sha
    return snapshot_download(REPO,revision=commit,token=False),commit

class FrozenEmotion2VecSeed:
    class_names=CLASSES
    def __init__(self,path,device='cpu'):
        from funasr import AutoModel
        import importlib.metadata
        path=Path(path)
        if not (path/'model.pt').is_file():raise FileNotFoundError(f'Expected downloaded seed model.pt: {path}')
        self.provenance=dict(repo=REPO,files={str(p.relative_to(path)):sha256(p) for p in sorted(path.rglob('*')) if p.is_file()},
                             funasr=importlib.metadata.version('funasr'))
        self.model=AutoModel(model=str(path.resolve()),device=device,disable_update=True)
        self.model.model.eval().requires_grad_(False)
    def probabilities(self,wave):
        import torch
        signal=np.asarray(wave,dtype=np.float32)
        if signal.ndim!=1 or not len(signal) or not np.isfinite(signal).all():raise ValueError('Invalid 16k mono input')
        with torch.inference_mode():
            result=self.model.generate(input=signal,granularity='utterance',extract_embedding=False)
        if len(result)!=1:raise ValueError('Expected one window result')
        return ordered_probabilities(result[0])
