"""Run identity, atomic commits, and fixed evaluation rules."""
from pathlib import Path
import hashlib
import json
import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from scripts.artifact_emotion_gated_attetnion_optimized.progress import Progress, status
from scripts.artifact_emotion_gated_attetnion_optimized.storage import WriterLock

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOURCE = ROOT / 'reports/artifact_emotion_gated_optimized_v1/cache'
OUTPUT = ROOT / 'reports/hierarchical_score_fusion_gru_temporal_attention_fav_v1/seed_42'
ARTIFACTS = ROOT / 'checkpoints/experiments/x3a_reproduction/fav_indomain_v1/unified_4gpu/x3a_4gpu_001'
DIMS = {'video': 1280, 'audio': 768}
CLASSES = {'video': ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise'],
           'audio': ['angry','disgusted','fearful','happy','neutral','other','sad','surprised','unknown']}

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def save_torch(path,value):
    path=Path(path);temp=path.with_name(path.name+'.partial')
    with temp.open('wb') as f: torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def implementation():
    files=[BASE/n for n in ('common.py','data.py','extract.py','model.py','train.py','experiment.py')]
    files += [ROOT/n for n in ('src/models/hsemotion.py','src/models/hierarchical_score_fusion_variation/emotion2vec_seed.py',
        'src/datasets/media.py','src/utils/local_paths.py','scripts/artifact_emotion_gated_attetnion_optimized/common.py',
        'scripts/artifact_emotion_gated_attetnion_optimized/local_inputs.py','scripts/artifact_emotion_gated_attetnion_optimized/storage.py',
        'scripts/artifact_emotion_gated_attetnion_optimized/progress.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    # cuDNN GRU may require CUBLAS_WORKSPACE_CONFIG on CUDA; the CLI sets it before importing torch.
    torch.use_deterministic_algorithms(True)

def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def score_or(a,b):return 1-(1-a)*(1-b)

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    require(y.shape==p.shape and np.isin(y,[0,1]).all() and np.isfinite(p).all(),'Invalid metric input')
    require(((p>=0)&(p<=1)).all(),'Score outside [0,1]')
    cm=confusion_matrix(y,p>=.5,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel());two=tn+fp>0 and fn+tp>0
    return dict(n=len(y),real=tn+fp,fake=fn+tp,roc_auc=float(roc_auc_score(y,p)) if two else None,
        average_precision=float(average_precision_score(y,p)) if two else None,
        balanced_accuracy=.5*(tn/(tn+fp)+tp/(tp+fn)) if two else None,
        real_fpr=fp/(tn+fp) if tn+fp else None,confusion_matrix=cm.tolist(),threshold=.5)
