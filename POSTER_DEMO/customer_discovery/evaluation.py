from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list

def classification_metrics(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> Metrics:
    try:
        roc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr = float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        pr_auc=pr,
        confusion_matrix=cm,
    )
