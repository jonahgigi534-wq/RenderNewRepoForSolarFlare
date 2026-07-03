"""Experiment 4 (the fix, decomposed): WHICH threshold recovers operational
skill — and is live data the active ingredient?

The deployed default threshold (0.104, "balanced") was tuned to maximise F1 on
the benchmark validation split. Two candidate fixes are frozen IN ADVANCE and
tested on operational years neither threshold ever saw:

  A. benchmark-val TSS threshold: re-tune the SAME benchmark validation split
     for TSS instead of F1 (the model's own stored high_recall operating point).
     Uses NO operational data at all.
  B. live-recalibrated threshold: the max-TSS threshold chosen on live JSOC
     2014 Q1 only, then frozen.

Both are evaluated at their fixed thresholds on live 2015 and live 2023.
If A ≈ B, the recovery is a THRESHOLD-OBJECTIVE effect — F1's precision term is
base-rate-sensitive and the operational base rate swings ~0.8%–4.3% across
years, while TSS is base-rate-insensitive — and live data is NOT the active
ingredient. CIs are cluster-bootstrap by active region (i.i.d. row resampling
would be dishonestly narrow for overlapping windows).
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np

from _common import (ROOT, RESULTS, load_or_build, bootstrap_tss_ci,
                     bootstrap_paired_tss_gain_ci, label_gated_periods)

from solarflare import data as dataio, evaluate
from solarflare.config import load_config
from solarflare.scorecard import label_gate_status

OUT = os.path.join(RESULTS, "recalibration.json")

cfg = load_config()
payload = joblib.load(os.path.join(ROOT, "models", "flare_sharp_live_model.joblib"))
model = payload["model"]
default_thr = float(payload["threshold"])                       # val-F1 optimum
benchval_thr = float(payload["operating_points"]["high_recall"]["threshold"])  # val-TSS optimum

CAL = ("2014_solar_max", datetime(2014, 1, 1, tzinfo=timezone.utc), datetime(2014, 4, 1, tzinfo=timezone.utc))
TESTS = [
    ("2015_declining",  datetime(2015, 6, 1, tzinfo=timezone.utc), datetime(2015, 9, 1, tzinfo=timezone.utc)),
    ("2017_declining",  datetime(2017, 8, 1, tzinfo=timezone.utc), datetime(2017, 11, 1, tzinfo=timezone.utc)),
    # Declared but excluded by the fail-closed label gate (HEK attribution 0.15
    # in 2023); the exclusion + reason are recorded in the output JSON.
    ("2023_rising_max", datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 4, 1, tzinfo=timezone.utc)),
]

ok, reason = label_gate_status(cfg, CAL[1].year)
if not ok:
    raise SystemExit(f"calibration year {CAL[1].year} fails the label gate: {reason}")
TESTS, _excluded = label_gated_periods(cfg, TESTS)
for tag, yr, why in _excluded:
    print(f"EXCLUDED {tag}: {why}", flush=True)


def proba_for(tag, t0, t1):
    d = load_or_build(tag, t0, t1, cfg)
    y = np.asarray(d["y"])
    Xf = dataio.build_matrix(d["X3d"], cfg)
    return y, model.predict_proba(Xf)[:, 1], d["groups"]


# Fix B: calibrate on live 2014 and freeze.
y_cal, p_cal, _ = proba_for(*CAL)
recal_thr, cal_tss = evaluate.best_threshold(y_cal, p_cal, "tss")
recal_thr = float(recal_thr)
print(f"calibration (2014 Q1): n={len(y_cal)} positives={int(y_cal.sum())}", flush=True)
print(f"  default (benchmark-val F1) threshold:      {default_thr:.3f}", flush=True)
print(f"  benchmark-val TSS threshold (no live data): {benchval_thr:.3f}", flush=True)
print(f"  live-recalibrated threshold (frozen):       {recal_thr:.3f}  (TSS on 2014: {cal_tss:.3f})", flush=True)

# Test all three frozen thresholds on unseen periods.
result = {"calibration_period": CAL[0],
          "default_threshold": round(default_thr, 3),
          "default_threshold_objective": "max F1 on benchmark validation",
          "benchmark_val_tss_threshold": round(benchval_thr, 3),
          "benchmark_val_tss_source": "max TSS on benchmark validation (model's stored "
                                      "high_recall operating point; no operational data)",
          "recalibrated_threshold": round(recal_thr, 3),
          "tss_on_calibration_period": round(float(cal_tss), 3),
          "ci_method": "cluster bootstrap by active region, 1000 replicates",
          "excluded_tests": {tag: {"year": yr, "reason": why}
                             for tag, yr, why in _excluded},
          "tests": {}}
for tag, t0, t1 in TESTS:
    y, p, groups = proba_for(tag, t0, t1)
    rows = {}
    for label, thr in (("default", default_thr),
                       ("benchmark_val_tss", benchval_thr),
                       ("recalibrated", recal_thr)):
        r = evaluate.full_report(y, (p >= thr).astype(int))
        lo, hi = bootstrap_tss_ci(y, p, thr, groups)
        rows[label] = {"threshold": round(thr, 3), "tss": round(r["tss"], 3),
                       "tss_ci95": [lo, hi], "recall": round(r["recall"], 3),
                       "precision": round(r["precision"], 3), "hss": round(r["hss"], 3)}
        print(f"[{tag}] {label:18s} thr={thr:.3f}  TSS={r['tss']:.3f} [{lo},{hi}]  "
              f"recall={r['recall']:.3f}  precision={r['precision']:.3f}", flush=True)
    rows["tss_gain"] = round(rows["recalibrated"]["tss"] - rows["default"]["tss"], 3)
    # Paired gain CI: both thresholds scored on the SAME resampled regions, so
    # the shared noise cancels — the honest significance test for the gain.
    rows["tss_gain_ci95"] = list(bootstrap_paired_tss_gain_ci(
        y, p, default_thr, recal_thr, groups))
    # The live-data increment: what recalibrating on live 2014 buys OVER the
    # benchmark's own TSS threshold. ~0 => the fix is the objective, not the data.
    rows["live_increment_over_benchval"] = round(
        rows["recalibrated"]["tss"] - rows["benchmark_val_tss"]["tss"], 3)
    rows["live_increment_ci95"] = list(bootstrap_paired_tss_gain_ci(
        y, p, benchval_thr, recal_thr, groups))
    result["tests"][tag] = rows

json.dump(result, open(OUT, "w"), indent=2)
print(f"\nDONE -> {OUT}", flush=True)
