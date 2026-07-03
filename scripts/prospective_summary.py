"""Snapshot the growing PROSPECTIVE forecast record into a committed artifact.

The prospective record is the one evidence class a benchmark cannot fake: every
daily P(M+ within 24 h) forecast was issued BEFORE its outcome was knowable, then
verified against what the Sun actually did. `solarflare.notify.prospective_record`
computes this live from the notifier DB; this script derives the same summary from
the committed prediction_history.csv (the versioned source of truth) and writes
prospective_record.json, so the record COMPOUNDS in git as N grows — months of
pre-event forecasts by competition time.

Adds cluster-free percentile bootstrap CIs (daily forecasts on distinct days are
not region-correlated, so i.i.d. row resampling is honest here, unlike the
windowed operational scorecard).

    python scripts/prospective_summary.py            # writes prospective_record.json
    python scripts/prospective_summary.py --check     # self-check only, writes nothing
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "prediction_history.csv")
OUT = os.path.join(ROOT, "prospective_record.json")


def _materialized(actual_peak_class: str) -> int:
    """1 if an M- or X-class flare actually occurred in the window, else 0."""
    c = (actual_peak_class or "").strip()[:1].upper()
    return 1 if c in ("M", "X") else 0


def _brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def _tss(pairs, thr):
    tp = sum(1 for p, y in pairs if p >= thr and y)
    fp = sum(1 for p, y in pairs if p >= thr and not y)
    fn = sum(1 for p, y in pairs if p < thr and y)
    tn = sum(1 for p, y in pairs if p < thr and not y)
    tss = (tp / (tp + fn) - fp / (fp + tn)) if (tp + fn) and (fp + tn) else None
    return tss, {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _ci(pairs, stat, n=2000, seed=42):
    """Percentile 95% CI via i.i.d. bootstrap; None if a replicate is undefined."""
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if len(vals) < n * 0.5:                       # too many undefined replicates
        return None
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return [round(lo, 4), round(hi, 4)]


def summarize(rows, alert_p=0.5):
    """rows: iterable of dicts from prediction_history.csv. Returns the record."""
    pairs = [(float(r["p_M_24h"]), _materialized(r["actual_peak_class"]))
             for r in rows
             if r.get("kind") == "daily" and r.get("status") == "verified"
             and r.get("p_M_24h") and (r.get("actual_peak_class") or "").strip()]
    n = len(pairs)
    if n == 0:
        return {"available": False, "n_days": 0,
                "note": "no verified daily forecasts yet — grows one per day the notifier runs"}
    events = sum(y for _, y in pairs)
    base = events / n
    brier = _brier(pairs)
    brier_clim = _brier([(base, y) for _, y in pairs])
    tss, conf = _tss(pairs, alert_p)
    return {
        "available": True,
        "n_days": n,
        "events": events,
        "base_rate": round(base, 3),
        "brier": round(brier, 4),
        "brier_ci95": _ci(pairs, _brier),
        "brier_climatology": round(brier_clim, 4),
        "brier_skill_vs_climatology": round(1 - brier / brier_clim, 3) if brier_clim else None,
        "tss_at_alert_threshold": round(tss, 3) if tss is not None else None,
        "tss_ci95": _ci(pairs, lambda s: _tss(s, alert_p)[0]),
        "alert_threshold_p": alert_p,
        "confusion": conf,
        "note": "prospective — every forecast issued before its outcome window; "
                "CIs are i.i.d. bootstrap (daily forecasts are not region-correlated). "
                "TSS/CIs firm up as both flare and no-flare days accumulate.",
    }


def demo():
    """Self-check on a fixed synthetic case (assert-based, no framework)."""
    rows = [
        {"kind": "daily", "status": "verified", "p_M_24h": "0.9", "actual_peak_class": "M1.0"},
        {"kind": "daily", "status": "verified", "p_M_24h": "0.1", "actual_peak_class": "C2.0"},
        {"kind": "daily", "status": "pending",  "p_M_24h": "0.5", "actual_peak_class": ""},  # ignored
        {"kind": "alert", "status": "verified", "p_M_24h": "0.9", "actual_peak_class": "X1"},  # not daily
    ]
    r = summarize(rows)
    assert r["n_days"] == 2, r                      # only the two verified daily rows count
    assert r["events"] == 1 and r["base_rate"] == 0.5, r
    # Brier = ((0.9-1)^2 + (0.1-0)^2)/2 = (0.01+0.01)/2 = 0.01
    assert abs(r["brier"] - 0.01) < 1e-9, r
    assert _materialized("X1.1") == 1 and _materialized("C9.5") == 0 and _materialized("") == 0
    print("prospective_summary self-check OK")


if __name__ == "__main__":
    if "--check" in sys.argv:
        demo()
    else:
        demo()
        with open(CSV, newline="") as f:
            rec = summarize(list(csv.DictReader(f)))
        with open(OUT, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"wrote {OUT}: n_days={rec.get('n_days')} "
              f"brier={rec.get('brier')} bss={rec.get('brier_skill_vs_climatology')}")
