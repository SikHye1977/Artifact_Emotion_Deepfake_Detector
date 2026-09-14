"""Collect full natural-distribution predictions, validating identity coverage."""
import torch
from .metrics import binary_metrics


def validate_predictions(records, expected_ids):
    ids = [r['sample_id'] for r in records]
    expected = list(expected_ids)
    if len(set(ids)) != len(ids) or len(set(expected)) != len(expected):
        raise ValueError('Duplicate sample_id')
    if set(ids) != set(expected):
        raise ValueError('Missing or unexpected prediction sample_id')


@torch.no_grad()
def evaluate(model, loader, step, device, amp=False):
    model.eval(); records=[]
    for batch in loader:
        with torch.autocast(device.type, enabled=amp, dtype=torch.float16):
            out, y, ids = step(model,batch,device)
        if not torch.isfinite(out.logits).all():raise FloatingPointError('Nonfinite evaluation logits')
        for sid,label,logits,prob in zip(ids,y.cpu().tolist(),out.logits.float().cpu().tolist(),out.fake_probability.float().cpu().tolist()):
            records.append(dict(sample_id=sid,label=label,logits=logits,fake_probability=prob))
    validate_predictions(records,[r['sample_id'] for r in loader.dataset.rows])
    return records, binary_metrics([r['label'] for r in records],[r['fake_probability'] for r in records])
