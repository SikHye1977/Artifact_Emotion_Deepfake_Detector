"""Local changes only; invalid observations are never bridged."""
import numpy as np


def change_features(values, times, valid, *, kind, segments=None, max_gap=1.5):
    values=np.asarray(values,dtype=np.float64)
    times=np.asarray(times,dtype=np.float64)
    valid=np.asarray(valid,dtype=bool)
    dim=values.shape[1] if values.ndim==2 else 0
    if kind not in ('video','audio') or dim not in ((8,) if kind=='video' else (8,9)) or values.shape!=(len(times),dim) or valid.shape!=times.shape:
        raise ValueError('Invalid trajectory shape or kind')
    if not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError('Strictly increasing finite timestamps required')
    if not np.isfinite(values[valid]).all(): raise ValueError('Nonfinite valid feature')
    x=values.copy()
    if ((x[valid]<0)|(x[valid]>1)).any() or not np.allclose(x[valid].sum(1),1,atol=1e-5):
        raise ValueError('Require normalized emotion probabilities')
    dt=np.diff(times)
    edges=valid[:-1]&valid[1:]&(dt>1e-6)&(dt<=max_gap)
    if segments is not None:
        segments=np.asarray(segments)
        if segments.shape!=times.shape: raise ValueError('Invalid segment shape')
        edges &= (segments[:-1]==segments[1:])&(segments[:-1]>=0)
    left,right=x[:-1][edges],x[1:][edges]
    delta=right-left
    elapsed=dt[edges]
    rate=delta/elapsed[:,None]
    magnitude=.5*np.abs(delta).sum(1) / elapsed
    transitions=np.concatenate([rate,magnitude[:,None]],axis=1).astype(np.float32)
    # 3 summaries per channel: signed mean, std and 95th percentile absolute value.
    pooled=np.concatenate([transitions.mean(0),transitions.std(0),
                           np.quantile(np.abs(transitions),.95,axis=0)]).astype(np.float32) if len(transitions) else np.zeros(3*(dim+1),np.float32)
    return dict(features=pooled, transitions=transitions, edge_mask=edges,
                dt=elapsed.astype(np.float32), valid_pairs=int(edges.sum()))
