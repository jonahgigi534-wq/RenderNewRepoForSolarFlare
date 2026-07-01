"""Experiment 2 (PAPER.md section 9, Figure 3): distribution-shift diagnosis.

(A) Era shift: raw JSOC SHARP for the benchmark era (Feb 2011) vs the operational
    year (Feb 2015), two-sample KS-tested per feature.
(B) Messiness: non-finite (missing/bad) rate per feature in raw JSOC — the mess
    the SWAN-SF benchmark removes via KNN imputation.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from _common import RESULTS

from solarflare import sharpdata
from solarflare.config import load_config

OUT = os.path.join(RESULTS, "exp2_distribution.json")
cfg = load_config()
keys = cfg["sharp_live"]["keywords"]


def fetch_matrix(t0, t1, tag):
    print(f"[{tag}] fetching raw JSOC {t0.date()}..{t1.date()} ...", flush=True)
    rows = sharpdata.fetch_sharp(t0, t1, cfg, verbose=True)
    M = np.stack([r[3] for r in rows]) if rows else np.empty((0, len(keys)))
    print(f"[{tag}] {len(M)} records", flush=True)
    return M


# One month each gives tens of thousands of records — ample for a KS test.
A = fetch_matrix(datetime(2011, 2, 1, tzinfo=timezone.utc), datetime(2011, 3, 1, tzinfo=timezone.utc), "2011_benchmark_era")
B = fetch_matrix(datetime(2015, 2, 1, tzinfo=timezone.utc), datetime(2015, 3, 1, tzinfo=timezone.utc), "2015_operational")

result = {"n_2011": int(len(A)), "n_2015": int(len(B)), "features": {}}
n_sig = 0
print(f"\n{'feature':10s} {'KS_D':>6} {'p':>10} {'miss2011':>9} {'miss2015':>9}", flush=True)
for i, k in enumerate(keys):
    a, b = A[:, i], B[:, i]
    miss_a = float(np.mean(~np.isfinite(a))) if len(a) else 0.0
    miss_b = float(np.mean(~np.isfinite(b))) if len(b) else 0.0
    af, bf = a[np.isfinite(a)], b[np.isfinite(b)]
    D, p = (stats.ks_2samp(af, bf) if len(af) > 10 and len(bf) > 10
            else (float("nan"), float("nan")))
    n_sig += int(bool(p < 0.05))
    result["features"][k] = {"ks_D": round(float(D), 3), "p": float(p),
                             "miss_2011": round(miss_a, 4), "miss_2015": round(miss_b, 4)}
    print(f"{k:10s} {D:6.3f} {p:10.2e} {miss_a:9.3%} {miss_b:9.3%}", flush=True)

result["n_features_shifted_p05"] = n_sig
result["median_ks_D"] = round(float(np.nanmedian([f["ks_D"] for f in result["features"].values()])), 3)
result["mean_missing_2015"] = round(float(np.mean([f["miss_2015"] for f in result["features"].values()])), 4)
print(f"\nfeatures significantly shifted (p<0.05): {n_sig}/{len(keys)}", flush=True)
json.dump(result, open(OUT, "w"), indent=2)
print(f"saved -> {OUT}", flush=True)
