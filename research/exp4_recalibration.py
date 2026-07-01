"""Experiment 4 (the validated fix): operational recalibration by threshold
transfer.

The paper argues most of the benchmark->live skill loss is a calibration
artifact. This experiment VALIDATES the fix as a procedure, with no test-set
peeking:

  1. CALIBRATE: choose the decision threshold that maximises TSS on live JSOC
     2014 Q1 (the earliest operational period).
  2. FREEZE it.
  3. TEST: evaluate at that fixed threshold on live 2015 and live 2023 —
     periods the threshold never saw.

If live TSS at the transferred threshold materially beats the benchmark-tuned
default threshold on those unseen periods, operational recalibration is a real,
deployable correction (not a hindsight artifact).
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np

from _common import ROOT, RESULTS, load_or_build, bootstrap_tss_ci

from solarflare import data as dataio, evaluate
from solarflare.config import load_config

OUT = os.path.join(RESULTS, "recalibration.json")

cfg = load_config()
payload = joblib.load(os.path.join(ROOT, "models", "flare_sharp_live_model.joblib"))
model = payload["model"]
default_thr = float(payload["threshold"])

CAL = ("2014_solar_max", datetime(2014, 1, 1, tzinfo=timezone.utc), datetime(2014, 4, 1, tzinfo=timezone.utc))
TESTS = [
    ("2015_declining",  datetime(2015, 6, 1, tzinfo=timezone.utc), datetime(2015, 9, 1, tzinfo=timezone.utc)),
    ("2023_rising_max", datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 4, 1, tzinfo=timezone.utc)),
]


def proba_for(tag, t0, t1):
    d = load_or_build(tag, t0, t1, cfg)
    y = np.asarray(d["y"])
    Xf = dataio.build_matrix(d["X3d"], cfg)
    return y, model.predict_proba(Xf)[:, 1]


# 1-2. Calibrate on 2014 and freeze.
y_cal, p_cal = proba_for(*CAL)
recal_thr, cal_tss = evaluate.best_threshold(y_cal, p_cal, "tss")
recal_thr = float(recal_thr)
print(f"calibration (2014 Q1): n={len(y_cal)} positives={int(y_cal.sum())}", flush=True)
print(f"  default (benchmark-tuned) threshold: {default_thr:.3f}", flush=True)
print(f"  recalibrated threshold (frozen):     {recal_thr:.3f}  (TSS on 2014: {cal_tss:.3f})", flush=True)

# 3. Test the frozen threshold on unseen periods.
result = {"calibration_period": CAL[0],
          "default_threshold": round(default_thr, 3),
          "recalibrated_threshold": round(recal_thr, 3),
          "tss_on_calibration_period": round(float(cal_tss), 3),
          "tests": {}}
for tag, t0, t1 in TESTS:
    y, p = proba_for(tag, t0, t1)
    rows = {}
    for label, thr in (("default", default_thr), ("recalibrated", recal_thr)):
        r = evaluate.full_report(y, (p >= thr).astype(int))
        lo, hi = bootstrap_tss_ci(y, p, thr)
        rows[label] = {"threshold": round(thr, 3), "tss": round(r["tss"], 3),
                       "tss_ci95": [lo, hi], "recall": round(r["recall"], 3),
                       "precision": round(r["precision"], 3), "hss": round(r["hss"], 3)}
        print(f"[{tag}] {label:12s} thr={thr:.3f}  TSS={r['tss']:.3f} [{lo},{hi}]  "
              f"recall={r['recall']:.3f}  precision={r['precision']:.3f}", flush=True)
    rows["tss_gain"] = round(rows["recalibrated"]["tss"] - rows["default"]["tss"], 3)
    result["tests"][tag] = rows

json.dump(result, open(OUT, "w"), indent=2)
print(f"\nDONE -> {OUT}", flush=True)
