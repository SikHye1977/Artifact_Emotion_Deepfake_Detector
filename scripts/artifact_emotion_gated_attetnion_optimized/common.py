"""Immutable cache/model provenance and deterministic feature-only evaluation."""
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = ROOT / 'checkpoints/experiments/x3a_reproduction/fav_indomain_v1/unified_4gpu/x3a_4gpu_001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def implementation():
    paths = sorted(BASE.glob('*.py'))
    paths += [ROOT / p for p in ['src/models/x3d.py', 'src/models/aasist.py', 'src/models/hsemotion.py',
        'src/models/hierarchical_score_fusion_variation/emotion2vec_seed.py',
        'src/models/hierarchical_score_fusion/hierarchical_checkpoint_loading.py',
        'src/datasets/frozen.py', 'src/datasets/media.py', 'src/utils/local_paths.py']]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def load_cache(path, modality, split=None):
    path = Path(path)
    if not (path / 'COMPLETED.json').is_file():
        raise ValueError('Incomplete feature cache: ' + str(path))
    meta = read(path / 'manifest.json')
    if meta['format'] != 'artifact_emotion_embedding_v1' or meta['modality'] != modality:
        raise ValueError('Wrong cache type/modality')
    if split and meta['split'] != split:
        raise ValueError('Wrong cache split')
    if sha(path / 'features.npz') != meta['sha256'] or digest(meta['recipe']) != meta['recipe_sha256']:
        raise ValueError('Cache hash mismatch')
    with np.load(path / 'features.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    n = meta['n']
    if n < 1 or any(len(v) != n for v in data.values()) or len(set(data['sample_id'])) != n:
        raise ValueError('Empty/duplicate/inconsistent cache rows')
    if data['labels'].shape != (n, 3) or not np.isin(data['labels'], [-1, 0, 1]).all():
        raise ValueError('Invalid labels')
    for key in ('artifact', 'emotion'):
        if data[key].ndim != 2 or not np.isfinite(data[key]).all():
            raise ValueError('Nonfinite/invalid embeddings')
    if data['artifact'].shape[1] != meta['artifact_dim'] or data['emotion'].shape[1] != meta['emotion_dim']:
        raise ValueError('Embedding dimensions differ from manifest')
    if data['emotion_valid'].dtype != np.bool_ or data['emotion_valid'].shape != (n,):
        raise ValueError('Invalid emotion validity mask')
    if not np.isfinite(data['artifact_probability']).all() or not np.isin(data['emotion_valid'], [False, True]).all():
        raise ValueError('Invalid cache values')
    if ((data['artifact_probability'] < 0) | (data['artifact_probability'] > 1)).any():
        raise ValueError('Invalid artifact probability')
    return data, meta


def disjoint(a, b):
    for key in ('sample_id', 'group'):
        if set(a[key]) & set(b[key]):
            raise ValueError('Train/dev overlap in ' + key)


def metrics(labels, probabilities):
    y = np.asarray(labels); p = np.asarray(probabilities)
    if not len(y):
        return {'n': 0, 'roc_auc': None}
    cm = confusion_matrix(y, p >= .5, labels=[0, 1])
    return dict(n=len(y), real=int((y == 0).sum()), fake=int((y == 1).sum()),
                roc_auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)),
                confusion_matrix=cm.tolist(), real_fpr=float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else None)
