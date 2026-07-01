"""Experiment 5 (generalizability): does the benchmark-vs-operational gap
replicate on a SECOND model architecture?

The paper's headline results all use the deployed RandomForest. If the gap is a
property of the BENCHMARK (our claim), not of one model family, then a different
architecture trained identically must show the same pattern. We train a
**LightGBM** gradient-boosted model on the same SWAN-SF partition-1 training set
with the same protocol (region-disjoint chronological split, thresholds tuned on
validation only), then score it exactly like Exp 1: on its held-out benchmark
test split and on the three live-JSOC periods.

Run:  python research/exp5_second_model.py   (needs data/sharp_live/dataset_swansf_p1.npz)
"""
import copy
import json
import os
from datetime import datetime, timezone

import numpy as np

from _common import ROOT, RESULTS, load_or_build, bootstrap_tss_ci

from solarflare import data as dataio, evaluate, sharpdata, sharptrain
from solarflare.config import load_config

OUT = os.path.join(RESULTS, "second_model.json")
TRAIN_NPZ = os.path.join(ROOT, "data", "sharp_live", "dataset_swansf_p1.npz")

PERIODS = [
    ("2014_solar_max",  datetime(2014, 1, 1, tzinfo=timezone.utc), datetime(2014, 4, 1, tzinfo=timezone.utc)),
    ("2015_declining",  datetime(2015, 6, 1, tzinfo=timezone.utc), datetime(2015, 9, 1, tzinfo=timezone.utc)),
    ("2023_rising_max", datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 4, 1, tzinfo=timezone.utc)),
]

cfg = load_config()

# Train LightGBM with the IDENTICAL protocol the RandomForest used — only the
# candidate family differs.
cfg_lgbm = copy.deepcopy(cfg)
cfg_lgbm["sharp_live"]["candidates"] = ["lightgbm"]
print("Training LightGBM on the SWAN-SF benchmark training set ...", flush=True)
payload = sharptrain.train(TRAIN_NPZ, cfg_lgbm, save=False)
model = payload["model"]
bench = payload.get("metrics", {})
ops = payload.get("operating_points", {})
print(f"\nLightGBM benchmark test: TSS={bench.get('tss', 0):.3f} "
      f"recall={bench.get('recall', 0):.3f} precision={bench.get('precision', 0):.3f}", flush=True)

result = {"architecture": "lightgbm",
          "protocol": "identical to RandomForest (region-disjoint chronological "
                      "split of SWAN-SF p1; thresholds tuned on validation only)",
          "benchmark_test": {k: round(bench.get(k, 0), 3)
                              for k in ("tss", "recall", "precision", "hss")},
          "periods": {}}

for name, t0, t1 in PERIODS:
    d = load_or_build(name, t0, t1, cfg)
    X3d, y = d["X3d"], np.asarray(d["y"])
    Xf = dataio.build_matrix(X3d, cfg)
    proba = model.predict_proba(Xf)[:, 1]
    rec = {"n": int(len(y)), "positives": int(y.sum()), "by_operating_point": {}}
    for op_name, op in ops.items():
        thr = float(op["threshold"])
        r = evaluate.full_report(y, (proba >= thr).astype(int))
        lo, hi = bootstrap_tss_ci(y, proba, thr)
        rec["by_operating_point"][op_name] = {
            "threshold": round(thr, 3), "tss": round(r["tss"], 3), "tss_ci95": [lo, hi],
            "recall": round(r["recall"], 3), "precision": round(r["precision"], 3)}
        print(f"[{name}] {op_name:14s} thr={thr:.3f}  TSS={r['tss']:.3f} [{lo},{hi}]  "
              f"recall={r['recall']:.3f}  precision={r['precision']:.3f}", flush=True)
    result["periods"][name] = rec
    json.dump(result, open(OUT, "w"), indent=2)

# Side-by-side with the RandomForest's Exp 1 numbers.
rf_path = os.path.join(RESULTS, "multiperiod_rescore.json")
if os.path.exists(rf_path):
    rf = json.load(open(rf_path))
    print("\n=== Gap replication: benchmark TSS vs live TSS @ balanced/default ===", flush=True)
    print(f"{'period':18s} {'RF live':>8} {'LGBM live':>10}", flush=True)
    result["rf_comparison"] = {}
    for name, _, _ in PERIODS:
        rf_t = rf["periods"][name]["by_operating_point"]["balanced"]["tss"]
        lg_t = result["periods"][name]["by_operating_point"].get("balanced", {}).get("tss")
        result["rf_comparison"][name] = {"rf_live_tss": rf_t, "lgbm_live_tss": lg_t}
        print(f"{name:18s} {rf_t:8.3f} {lg_t if lg_t is None else format(lg_t, '10.3f')}", flush=True)
    result["rf_benchmark_tss"] = rf["benchmark_swansf"]["tss"]

json.dump(result, open(OUT, "w"), indent=2)
print(f"\nDONE -> {OUT}", flush=True)
