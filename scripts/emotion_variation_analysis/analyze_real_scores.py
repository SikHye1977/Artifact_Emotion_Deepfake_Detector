"""Read saved predictions only. No inference, fitting or threshold selection."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

SCORES = ('artifact_only', 'video_emotion', 'audio_emotion',
          'emotion_only', 'hierarchical')


def prob_or(a, b):
    return 1 - (1-a)*(1-b)


def load_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'sample_id', 'included', 'clip_label', 'video_label',
                    'audio_label', *SCORES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {required-set(reader.fieldnames or [])}')
        rows = list(reader)
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample_id')
    for r in rows:
        flag = r['included'].lower()
        if flag not in ('true', 'false', '1', '0'):
            raise ValueError('Invalid inclusion flag')
        r['included'] = flag in ('true', '1')
        for key in ('clip_label', 'video_label', 'audio_label'):
            r[key] = int(r[key])
            if r[key] not in (-1, 0, 1):
                raise ValueError('Invalid label')
        if not r['included']:
            continue
        if r['clip_label'] not in (0, 1):
            raise ValueError('Unknown included label')
        for key in SCORES:
            r[key] = float(r[key])
            if not math.isfinite(r[key]) or not 0 <= r[key] <= 1:
                raise ValueError(f'Invalid score: {key}')
        if not math.isclose(r['emotion_only'], prob_or(r['video_emotion'], r['audio_emotion']), abs_tol=2e-6):
            raise ValueError('Emotion OR mismatch')
        if not math.isclose(r['hierarchical'], prob_or(r['artifact_only'], r['emotion_only']), abs_tol=2e-6):
            raise ValueError('Final OR mismatch')
    return rows


def summarize(rows, expected_real=None):
    kept = [r for r in rows if r['included']]
    real = [dict(r) for r in kept if r['clip_label'] == 0]
    if not real or (expected_real is not None and len(real) != expected_real):
        raise ValueError(f'Unexpected real count: {len(real)} (expected {expected_real})')
    if any(r['video_label'] != 0 or r['audio_label'] != 0 for r in real):
        raise ValueError('Real clip has inconsistent modality labels')
    for r in real:
        v, a = r['video_emotion'] >= .5, r['audio_emotion'] >= .5
        r['emotion_pattern'] = ('both_high' if v and a else 'video_only_high' if v
                                else 'audio_only_high' if a else
                                'both_low_or_high' if r['emotion_only'] >= .5 else 'both_low_or_low')
        r['artifact_plus_video'] = prob_or(r['artifact_only'], r['video_emotion'])
        r['artifact_plus_audio'] = prob_or(r['artifact_only'], r['audio_emotion'])
        r['harmed'] = r['artifact_only'] < .5 <= r['hierarchical']
        r['final_threshold_accumulation'] = (r['artifact_only'] < .5 and
                                            r['emotion_only'] < .5 <= r['hierarchical'])
        r['delta_remove_video'] = r['hierarchical'] - r['artifact_plus_audio']
        r['delta_remove_audio'] = r['hierarchical'] - r['artifact_plus_video']
        r['video_removal_corrects'] = r['hierarchical'] >= .5 > r['artifact_plus_audio']
        r['audio_removal_corrects'] = r['hierarchical'] >= .5 > r['artifact_plus_video']
    names = (*SCORES, 'artifact_plus_video', 'artifact_plus_audio')
    stats = {}
    for key in names:
        values = [r[key] for r in real]
        fp = sum(x >= .5 for x in values)
        stats[key] = dict(n=len(values), fp=fp, tn=len(values)-fp,
                          fpr=fp/len(values), specificity=1-fp/len(values),
                          mean=mean(values), median=median(values),
                          minimum=min(values), maximum=max(values))
    ablation = {}
    for name in ('artifact_only', 'artifact_plus_video', 'artifact_plus_audio', 'hierarchical'):
        tn = fp = fn = tp = rescued = harmed = 0
        for r in kept:
            score = (prob_or(r['artifact_only'], r['video_emotion']) if name == 'artifact_plus_video'
                     else prob_or(r['artifact_only'], r['audio_emotion']) if name == 'artifact_plus_audio'
                     else r[name])
            y, pred, base = r['clip_label'], score >= .5, r['artifact_only'] >= .5
            tn += y == 0 and not pred
            fp += y == 0 and pred
            fn += y == 1 and not pred
            tp += y == 1 and pred
            rescued += base != y and pred == y
            harmed += base == y and pred != y
        ablation[name] = dict(n=len(kept), tn=tn, fp=fp, fn=fn, tp=tp,
                              rescued_vs_artifact=rescued, harmed_vs_artifact=harmed)
    summary = dict(threshold=.5, total=len(rows), common_valid=len(kept), real_n=len(real),
                   excluded_real=sum(not r['included'] and r['clip_label'] == 0 for r in rows),
                   real_score_statistics=stats,
                   emotion_patterns=dict(Counter(r['emotion_pattern'] for r in real)),
                   harmed_patterns=dict(Counter(r['emotion_pattern'] for r in real if r['harmed'])),
                   final_threshold_accumulation=sum(r['final_threshold_accumulation'] for r in real),
                   common_set_ablations=ablation,
                   interpretation='Descriptive fixed-score ablation, not causal attribution. Real-only AUC/BA not computed. No threshold tuning.')
    return sorted(real, key=lambda r: r['hierarchical'], reverse=True), summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-real', type=int)
    args = p.parse_args()
    rows = load_rows(args.predictions)
    real, summary = summarize(rows, args.expected_real)
    summary['source'] = str(args.predictions.resolve())
    summary['source_sha256'] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'real_scores.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(real[0]))
        w.writeheader()
        w.writerows(real)
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
