"""Atomic per-sample commits and OS-held writer lock (released on process death)."""
from pathlib import Path
import os
import json
import numpy as np
from .common import sha,digest,read


def atomic_json(path, value):
    path=Path(path);temp=path.with_name(path.name+'.partial')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def atomic_npz(path, **arrays):
    path=Path(path);temp=path.with_name(path.name+'.partial')
    with temp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


class WriterLock:
    def __init__(self, root):self.path=Path(root)/'.writer.lock'
    def __enter__(self):
        self.file=self.path.open('a+b');self.file.seek(0);self.file.write(b'0');self.file.flush();self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close();raise RuntimeError('Another extractor is writing this output')
        return self
    def __exit__(self,*args):
        self.file.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.file,fcntl.LOCK_UN)
        self.file.close()


class SampleStore:
    def __init__(self,root,recipe,split,rows,resume):
        self.root=Path(root);self.folder=self.root/'samples';self.recipe=recipe
        self.folder.mkdir(exist_ok=True)
        path=self.root/'STARTED.json'
        descriptor=dict(format='gated_resumable_v1',recipe=recipe,split=split,
                        rows_sha256=digest([(r['sample_id'],r['group'],r['labels'],r['path']) for r in rows]))
        if path.exists():
            if not resume:raise FileExistsError('Use --resume for this resumable output')
            if read(path)!=descriptor:raise ValueError('Resume recipe/split/row order changed')
        else:
            if list(self.folder.iterdir()):raise ValueError('Sample files without run descriptor')
            atomic_json(path,descriptor)
    def load(self,row):
        path=self.folder/(digest(row['sample_id'])+'.npz');marker=path.with_suffix('.json')
        if not path.exists() or not marker.exists():return None
        try:
            meta=read(marker)
            if sha(path)!=meta['sha256']:return None
            stat=Path(row['path']).stat()
            if meta['size']!=stat.st_size or meta['mtime_ns']!=stat.st_mtime_ns:raise ValueError('Local media changed since saved sample')
            with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
            if str(data['sample_id'])!=row['sample_id']:raise ValueError('Saved sample ID mismatch')
            return data
        except (OSError,EOFError,json.JSONDecodeError):return None
    def save(self,row,artifact,probability,tokens,pooled):
        stat=Path(row['path']).stat();path=self.folder/(digest(row['sample_id'])+'.npz')
        payload=dict(sample_id=np.asarray(row['sample_id']),artifact=np.asarray(artifact,np.float32),
            artifact_probability=np.asarray(probability),emotion=np.asarray(pooled,np.float32),emotion_valid=np.asarray(tokens['valid'].any()))
        payload.update({'token_'+k:np.asarray(v) for k,v in tokens.items()})
        atomic_npz(path,**payload)
        atomic_json(path.with_suffix('.json'),dict(sha256=sha(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
        return payload
