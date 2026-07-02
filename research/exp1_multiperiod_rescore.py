"""Experiment 1 (PAPER.md section 5, Figure 1): multi-period live-JSOC re-score
of the SWAN-SF-trained model.

Scores the deployed model on three out-of-sample periods spanning different
solar-cycle phases, via the same JSOC+HEK -> build_matrix transform the live
system uses. Reports TSS/recall/precision at each operating point with
cluster-bootstrap 95% CIs (whole active regions resampled — windows of one
region overlap in time, so i.i.d. row resampling would be dishonestly narrow).
All periods postdate the training span (2010-05 .. 2012-03).
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np

from _common import ROOT, RESULTS, load_or_build, bootstrap_tss_ci

from solarflare import data as dataio, evaluate
from solarflare.config import load_config

PERIODS = [
    ("2014_solar_max",  datetime(2014, 1, 1, tzinfo=timezone.utc), datetime(2014, 4, 1, tzinfo=timezone.utc)),
    ("2015_declining",  datetime(2015, 6, 1, tzinfo=timezone.utc), datetime(2015, 9, 1, tzinfo=timezone.utc)),
    ("2023_rising_max", datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 4, 1, tzinfo=timezone.utc)),
]
OUT = os.path.join(RESULTS, "multiperiod_rescore.json")

cfg = load_config()
payload = joblib.load(os.path.join(ROOT, "models", "flare_sharp_live_model.joblib"))
model = payload["model"]
ops = payload.get("operating_points", {})

results = {"training_span": payload.get("data_span"),
           "benchmark_swansf": {k: round(payload.get("metrics", {}).get(k, 0), 3)
                                 for k in ("tss", "recall", "precision", "hss")},
           "periods": {}}

for name, t0, t1 in PERIODS:
    print(f"\n===== {name}: {t0.date()} .. {t1.date()} =====", flush=True)
    d = load_or_build(name, t0, t1, cfg)
    X3d, y, groups = d["X3d"], np.asarray(d["y"]), d["groups"]
    n, pos = len(y), int(y.sum())
    print(f"  windows={n} positives={pos} base_rate={(pos/n if n else 0):.4f}", flush=True)
    if n == 0 or pos == 0:
        results["periods"][name] = {"n": n, "positives": pos, "note": "no positives"}
        continue
    Xf = dataio.build_matrix(X3d, cfg)
    proba = model.predict_proba(Xf)[:, 1]
    rec = {"n": n, "positives": pos, "base_rate": round(pos / n, 4), "by_operating_point": {}}
    for op_name, op in ops.items():
        thr = float(op["threshold"])
        r = evaluate.full_report(y, (proba >= thr).astype(int))
        lo, hi = bootstrap_tss_ci(y, proba, thr, groups)
        rec["by_operating_point"][op_name] = {
            "threshold": round(thr, 3),
            "tss": round(r["tss"], 3), "tss_ci95": [lo, hi],
            "recall": round(r["recall"], 3), "precision": round(r["precision"], 3),
            "hss": round(r["hss"], 3),
        }
        print(f"  {op_name:14s} thr={thr:.3f}  TSS={r['tss']:.3f} [{lo},{hi}]  "
              f"recall={r['recall']:.3f}  precision={r['precision']:.3f}", flush=True)
    results["periods"][name] = rec
    json.dump(results, open(OUT, "w"), indent=2)          # incremental save

json.dump(results, open(OUT, "w"), indent=2)
print(f"\nDONE -> {OUT}", flush=True)
