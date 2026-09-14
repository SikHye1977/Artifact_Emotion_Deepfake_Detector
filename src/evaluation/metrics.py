"""Binary metrics with explicit AP definition and positive class 1."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix)


def binary_metrics(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    if y.ndim != 1 or p.shape != y.shape or set(y.tolist()) != {0, 1}:
        raise ValueError('Aligned binary labels containing both classes required')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid fake probabilities')
    pred = p >= .5
    return dict(roc_auc=float(roc_auc_score(y,p)), average_precision=float(average_precision_score(y,p)),
        accuracy_at_0_5=float(accuracy_score(y,pred)), balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)), mcc=float(matthews_corrcoef(y,pred)),
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
