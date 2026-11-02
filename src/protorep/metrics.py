"""Detection metrics: AUC-PR, AUC-ROC, best-F1 with and without point adjustment."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def point_adjust(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """If any point of a true anomaly segment is flagged, flag the whole segment."""
    pred = pred.copy().astype(bool)
    y = y.astype(bool)
    i = 0
    n = len(y)
    while i < n:
        if y[i]:
            j = i
            while j < n and y[j]:
                j += 1
            if pred[i:j].any():
                pred[i:j] = True
            i = j
        else:
            i += 1
    return pred


def f1(pred: np.ndarray, y: np.ndarray) -> float:
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def best_f1(score: np.ndarray, y: np.ndarray, adjust: bool = False, n_thr: int = 200) -> float:
    qs = np.unique(np.quantile(score, np.linspace(0.0, 0.999, n_thr)))
    best = 0.0
    for t in qs:
        p = score > t
        if adjust:
            p = point_adjust(p, y)
        best = max(best, f1(p, y))
    return best


def detection_report(score: np.ndarray, y: np.ndarray, thr: float | None = None) -> dict:
    score = np.nan_to_num(np.asarray(score, float), nan=0.0, posinf=1e9)
    out = dict(auc_pr=float(average_precision_score(y, score)),
               auc_roc=float(roc_auc_score(y, score)),
               f1_best=best_f1(score, y, False),
               f1_best_pa=best_f1(score, y, True))
    if thr is not None:
        p = score > thr
        out["f1_ref_thr"] = f1(p, y)
        out["f1_ref_thr_pa"] = f1(point_adjust(p, y), y)
    return out
