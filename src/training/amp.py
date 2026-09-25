"""Post-unscale gradient diagnosis and recoverable AMP overflow policy."""
from collections import Counter
import torch


class GradientPolicy:
    def __init__(self, min_scale=1., max_consecutive=3, required=(), clip_norm=None):
        self.min_scale=min_scale; self.max_consecutive=max_consecutive
        self.required=set(required); self.clip_norm=clip_norm
        self.consecutive=0; self.combinations=Counter(); self.history=[]

    def finish(self, model, optimizer, scaler, sample_ids):
        old=float(scaler.get_scale())
        scaler.unscale_(optimizer)
        named={n:p for n,p in model.named_parameters() if p.requires_grad}
        missing=[n for n,p in named.items() if p.grad is None]
        nonfinite=[n for n,p in named.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
        finite=[n for n,p in named.items() if p.grad is not None and torch.isfinite(p.grad).all()]
        record=dict(previous_scale=old,missing_gradient=missing,nonfinite_gradient=nonfinite,
                    finite_gradient_parameters=len(finite),step_performed=False)
        self.history.append(record)
        required_missing=sorted(self.required-set(named) | (self.required & set(missing)))
        # A GradScaler overflow may make every participating gradient inf. It
        # is recoverable while AMP is enabled; only missing required grads or
        # a graph with no gradients at all are immediately fatal.
        if required_missing or (not finite and not nonfinite):
            record['fatal']='required_missing_or_no_finite_gradient'
            raise FloatingPointError(str(record))
        overflow=bool(nonfinite)
        if overflow and not scaler.is_enabled():
            record['fatal']='nonfinite_fp32_gradient'
            raise FloatingPointError(str(record))
        if not overflow and self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(named.values(),self.clip_norm,error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        step_performed=not overflow
        if step_performed:
            post_step_nonfinite=[n for n,p in named.items() if not torch.isfinite(p).all()]
            if post_step_nonfinite:
                record['post_step_nonfinite_parameters']=post_step_nonfinite
                record['fatal']='nonfinite_parameter_after_successful_step'
                raise FloatingPointError(str(record))
        record.update(new_scale=float(scaler.get_scale()),step_performed=step_performed,overflow=overflow)
        self.consecutive=self.consecutive+1 if overflow else 0
        if overflow:self.combinations[tuple(sorted(sample_ids))]+=1
        reason=None
        if self.consecutive>=self.max_consecutive:reason='consecutive_overflow'
        if overflow and self.combinations[tuple(sorted(sample_ids))]>=2:reason='repeated_sample_combination_overflow'
        if record['new_scale']<self.min_scale:reason='scale_below_minimum'
        if reason:
            record['fatal']=reason
            raise FloatingPointError(reason)
        return record

    def state_dict(self):
        return dict(consecutive=self.consecutive,combinations=dict(self.combinations))

    def load_state_dict(self,state):
        self.consecutive=state['consecutive'];self.combinations=Counter(state['combinations'])
