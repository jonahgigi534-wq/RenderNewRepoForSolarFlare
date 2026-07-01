"""Shared plumbing for the research scripts: repo-root resolution, dataset
caching (fetch once from JSOC/HEK, reuse thereafter), and bootstrap TSS CIs."""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "research", "results")
FIGURES = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data", "sharp_live")
os.makedirs(RESULTS, exist_ok=True)


def load_or_build(tag: str, t0: datetime, t1: datetime, cfg: dict) -> dict:
    """Cached JSOC+HEK dataset for [t0, t1): data/sharp_live/eval_<tag>.npz."""
    from solarflare import sharpdata
    path = os.path.join(DATA, f"eval_{tag}.npz")
    if os.path.exists(path):
        print(f"[{tag}] cached -> {path}", flush=True)
        return sharpdata.load_dataset(path)
    print(f"[{tag}] building from JSOC+HEK {t0.date()}..{t1.date()} ...", flush=True)
    d = sharpdata.build_dataset(t0, t1, cfg, verbose=True)
    sharpdata.save_dataset(d, path)
    return d


def tss_from(y, pred) -> float:
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    return recall - far


def bootstrap_tss_ci(y, proba, thr, n=1000, seed=42):
    """Percentile-bootstrap 95% CI on TSS at a fixed threshold."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); proba = np.asarray(proba)
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        vals.append(tss_from(y[s], (proba[s] >= thr).astype(int)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)
