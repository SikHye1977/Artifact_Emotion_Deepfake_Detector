"""Descriptive fixed-threshold reports; no test-time model/threshold selection."""
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve,precision_recall_curve

def csv_rows(path, rows):
    rows=list(rows)
    if not rows:return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def export_analysis(data,scores,keep,report,output,split):
    indices=np.flatnonzero(keep);lookup={int(i):j for j,i in enumerate(indices)}
    rows=[]
    for i,sid in enumerate(data['sample_id']):
        row=dict(sample_id=str(sid),group=str(data['group'][i]),clip_label=int(data['labels'][i,0]),
                 video_label=int(data['labels'][i,1]),audio_label=int(data['labels'][i,2]),
                 video_pairs=int(data['video_count'][i]),audio_pairs=int(data['audio_count'][i]),included=bool(keep[i]),
                 exclusion=';'.join(k for k,c in [('no_video_pairs',data['video_count'][i]==0),('no_audio_pairs',data['audio_count'][i]==0),('unknown_clip_label',data['labels'][i,0]<0)] if c))
        j=lookup.get(i)
        for name,values in scores.items():row[name]=float(values[j]) if j is not None else ''
        if j is None:row['outcome']='excluded'
        else:
            truth=int(data['labels'][i,0]);a=scores['artifact_only'][j]>=.5;h=scores['hierarchical'][j]>=.5
            row['outcome']='rescued' if a!=truth and h==truth else 'harmed' if a==truth and h!=truth else 'both_correct' if h==truth else 'both_wrong'
        rows.append(row)
    csv_rows(output/f'{split}_predictions.csv',rows)
    csv_rows(output/f'{split}_errors_and_changes.csv',[r for r in rows if r['outcome']!='both_correct'])
    csv_rows(output/f'{split}_metrics.csv',[dict(model=k,**v) for k,v in report.items() if isinstance(v,dict) and 'roc_auc' in v])
    y=data['labels'][keep,0]
    if len(np.unique(y))==2:
        for name in ['artifact_only','emotion_only','hierarchical']:
            fpr,tpr,th=roc_curve(y,scores[name]);precision,recall,threshold=precision_recall_curve(y,scores[name])
            csv_rows(output/f'{split}_{name}_roc.csv',[dict(fpr=float(f),tpr=float(t),threshold=float(v)) for f,t,v in zip(fpr,tpr,th)])
            csv_rows(output/f'{split}_{name}_pr.csv',[dict(precision=float(p),recall=float(r)) for p,r in zip(precision,recall)])
    return dict(rescued=sum(r['outcome']=='rescued' for r in rows),harmed=sum(r['outcome']=='harmed' for r in rows),
                additional_fake_detected=sum(r['outcome']=='rescued' and r['clip_label']==1 for r in rows),
                additional_real_false_positives=sum(r['outcome']=='harmed' and r['clip_label']==0 for r in rows))
