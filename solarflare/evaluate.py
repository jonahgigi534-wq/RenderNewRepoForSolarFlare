"""Flare-forecasting skill scores.

Plain accuracy is meaningless for flares: a model that always says "no-flare"
scores ~95% because flares are rare. The field uses skill scores that reward
catching the rare positives:

    TSS  (True Skill Statistic) = recall - false_alarm_rate
                                = TP/(TP+FN) - FP/(FP+TN)
         Ranges -1..1; 0 = no skill (random/always-one-class). This is the
         headline metric and what we tune the decision threshold against.

    HSS  (Heidke Skill Score) = improvement over random chance, accounting
         for class imbalance. > 0 means skill.

We also report recall (how many flares we caught), precision (how often a
warning was right), and the full confusion matrix.
"""
from __future__ import annotations

import numpy as np


def confusion_binary(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def tss(y_true, y_pred) -> float:
    tp, tn, fp, fn = confusion_binary(y_true, y_pred)
    recall = tp / (tp + fn) if (tp + fn) else 0.0          # sensitivity
    far = fp / (fp + tn) if (fp + tn) else 0.0             # false-alarm rate
    return recall - far


def hss(y_true, y_pred) -> float:
    tp, tn, fp, fn = confusion_binary(y_true, y_pred)
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0
    expected = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n
    denom = n - expected
    return (tp + tn - expected) / denom if denom else 0.0


def full_report(y_true, y_pred) -> dict:
    tp, tn, fp, fn = confusion_binary(y_true, y_pred)
    total = tp + tn + fp + fn
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": total,
        "positives": tp + fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "recall": recall,                 # flares caught
        "precision": precision,           # warning reliability
        "specificity": specificity,
        "f1": f1,
        "tss": tss(y_true, y_pred),       # headline skill score
        "hss": hss(y_true, y_pred),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def precision_recall_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tp, tn, fp, fn = confusion_binary(y_true, pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def fbeta(y_true, pred, beta=1.0):
    tp, tn, fp, fn = confusion_binary(y_true, pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def operating_points(y_true, proba, min_recall=0.25):
    """Three thresholds fitted on validation, each a different trade-off.

    high_recall    : max TSS  -> catches the most flares (low threshold)
    balanced       : max F1   -> even precision/recall (mid threshold)
    high_precision : MAX precision subject to recall >= min_recall
                     -> fewest false alarms (high threshold)
    """
    t_recall, _ = best_threshold(y_true, proba, "tss")
    t_balanced, _ = best_threshold(y_true, proba, "f1")

    # High precision: among thresholds keeping recall >= floor, take the most
    # precise. If the floor is unreachable, fall back to max F0.5.
    thresholds = np.unique(np.round(proba, 3))
    t_precision, best_prec = None, -1.0
    for t in thresholds:
        prec, rec = precision_recall_at(y_true, proba, t)
        if rec >= min_recall and prec > best_prec:
            best_prec, t_precision = prec, float(t)
    if t_precision is None:
        best_f = -1.0
        for t in thresholds:
            f = fbeta(y_true, (proba >= t).astype(int), beta=0.5)
            if f > best_f:
                best_f, t_precision = f, float(t)

    out = {}
    for name, t in (("high_recall", t_recall), ("balanced", t_balanced),
                    ("high_precision", t_precision)):
        rep = full_report(y_true, (proba >= t).astype(int))
        out[name] = {"threshold": float(t), "val": rep}
    return out


def best_threshold(y_true, proba, metric="tss"):
    """Scan thresholds and pick the one maximising the chosen skill score.

    This is how we honour ">50% / weighted" performance without cheating: the
    threshold is fit on the VALIDATION set and then frozen for the test set.
    """
    score_fn = {
        "tss": tss,
        "hss": hss,
        "f1": lambda yt, yp: fbeta(yt, yp, 1.0),
    }[metric]
    thresholds = np.unique(np.round(proba, 3))
    best_t, best_s = 0.5, -2.0
    for t in thresholds:
        s = score_fn(y_true, (proba >= t).astype(int))
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s


def pretty(report: dict) -> str:
    c = report["confusion"]
    return (
        f"  samples={report['n']}  positives={report['positives']}\n"
        f"  TSS={report['tss']:.3f}   HSS={report['hss']:.3f}   "
        f"F1={report['f1']:.3f}\n"
        f"  recall(flares caught)={report['recall']:.3f}   "
        f"precision={report['precision']:.3f}   "
        f"specificity={report['specificity']:.3f}\n"
        f"  confusion  TP={c['tp']} FN={c['fn']} | FP={c['fp']} TN={c['tn']}"
    )
