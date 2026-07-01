"""Does the STORM model show the same benchmark optimism as the flare model?

Generalisation check for the research question: train the geomagnetic-storm
model with the standard benchmark protocol (chronological split inside a fixed
OMNI era), then score it on a LATER, completely unseen era — the operational
analogue. If the benchmark-vs-operational gap appears here too, the finding is
not a flare-model quirk; it generalises across space-weather forecasting tasks.

  * BENCHMARK score  = held-out chronological test slice inside the training era
                       (the number a paper would report).
  * OPERATIONAL score = a later era the model has never seen, from a different
                        phase of the solar cycle (what deployment actually faces).
  * Peak TSS (threshold-free ceiling) AND frozen TSS (validation-tuned threshold
    committed in advance) — same two views as the flare scorecard.
  * BLOCK bootstrap CIs: storm windows are 3-hourly and heavily autocorrelated,
    so we resample multi-day blocks, not single windows (i.i.d. resampling would
    make the CIs dishonestly narrow).

Trains in-memory only — the deployed storm model artifact is untouched.
Writes storm_scorecard.json (merged into /api/scorecard by the server).

Run:  python -m solarflare.storm_scorecard
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight

from . import data as dataio
from . import evaluate as ev
from . import sources, stormdata
from .config import load_config
from .scorecard import _ci, frozen_tss, peak_tss


def _era(cfg: dict, start: str, stop: str):
    """OMNI era -> (X features, y labels). Raises if OMNI is unreachable."""
    res = sources.get_omni(cfg, f"{start}T00:00:00Z", f"{stop}T00:00:00Z")
    if not res.ok or not res.data:
        raise RuntimeError(f"OMNI fetch failed for {start}..{stop}: {res.error or 'no data'}")
    times, channels, kp = stormdata.parse_omni_csv(res.data)
    if len(kp) < 1000:
        raise RuntimeError(f"OMNI returned too few rows ({len(kp)}) for {start}..{stop}")
    X3d, y, t = stormdata._windows(channels, kp, times, cfg)
    return dataio.build_matrix(X3d, stormdata._stats_cfg(cfg)), y, t


def block_indices(n: int, block: int, rng) -> np.ndarray:
    """One moving-block-bootstrap resample of 0..n-1 (blocks of `block` samples)."""
    out, total = [], 0
    hi = max(1, n - block + 1)
    while total < n:
        s = int(rng.integers(0, hi))
        seg = np.arange(s, min(s + block, n))
        out.append(seg)
        total += len(seg)
    return np.concatenate(out)[:n]


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    sc = cfg.get("storm_scorecard", {}) or {}
    bench_span = (str(sc.get("benchmark_start", "2011-01-01")),
                  str(sc.get("benchmark_end", "2021-01-01")))
    op_span = (str(sc.get("operational_start", "2021-01-01")),
               str(sc.get("operational_end", "2025-01-01")))
    B = int(sc.get("bootstrap_replicates", 500))
    level = float(sc.get("ci_level", 0.95))
    stride = int(cfg["storm"].get("sample_stride_h", 3))
    block = max(1, int(sc.get("block_days", 7)) * 24 // stride)   # ~7 days of windows
    rng = np.random.default_rng(int(sc.get("random_state", 42)))

    print(f"Benchmark era {bench_span[0]}..{bench_span[1]} (train/val/test inside) ...", flush=True)
    Xb, yb, tb = _era(cfg, *bench_span)
    print(f"  windows={len(yb)}  storm rate={yb.mean():.3f}", flush=True)
    print(f"Operational era {op_span[0]}..{op_span[1]} (fully unseen) ...", flush=True)
    Xo, yo, _ = _era(cfg, *op_span)
    print(f"  windows={len(yo)}  storm rate={yo.mean():.3f}", flush=True)

    (Xtr, ytr), (Xva, yva), (Xte, yte) = stormdata._chrono_split(Xb, yb, tb, cfg)
    if ytr.sum() == 0 or yva.sum() == 0 or yte.sum() == 0 or yo.sum() == 0:
        raise RuntimeError("a split has no positive storms — widen the eras")

    # Bake-off (mirrors storm.train_model, in-memory only).
    from . import storm as storm_mod
    sw = compute_sample_weight("balanced", ytr)
    results = []
    print("\n=== Bake-off (ranked on validation TSS) ===", flush=True)
    for name, est in storm_mod._build_candidates(cfg).items():
        t0 = time.time()
        try:
            try:
                est.fit(Xtr, ytr, sample_weight=sw)
            except (TypeError, ValueError):
                est.fit(Xtr, ytr)
            pv = est.predict_proba(Xva)[:, 1]
            thr_v, _ = ev.best_threshold(yva, pv, "tss")
            skill = ev.tss(yva, (pv >= thr_v).astype(int))
            results.append((name, skill))
            print(f"  {name:24s} val TSS={skill:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as exc:                              # noqa: BLE001
            print(f"  {name:24s} FAILED: {type(exc).__name__}", flush=True)
    if not results:
        raise RuntimeError("no storm candidate trained")
    results.sort(key=lambda r: r[1], reverse=True)
    winner = results[0][0]
    print(f"  -> winner: {winner}", flush=True)

    model = CalibratedClassifierCV(storm_mod._build_candidates(cfg)[winner],
                                   method="isotonic", cv=3)
    try:
        model.fit(Xtr, ytr, sample_weight=sw)
    except (TypeError, ValueError):
        model.fit(Xtr, ytr)

    # The pre-committed deployment threshold: validation-TSS-optimal.
    pva = model.predict_proba(Xva)[:, 1]
    thr, _ = ev.best_threshold(yva, pva, "tss")
    pte = model.predict_proba(Xte)[:, 1]
    po = model.predict_proba(Xo)[:, 1]

    # Block bootstrap both test sets (same replicate index -> paired gap).
    boot = {"benchmark": {"peak": np.empty(B), "frozen": np.empty(B)},
            "operational": {"peak": np.empty(B), "frozen": np.empty(B)}}
    print(f"Block bootstrap ({B} replicates, block={block} windows) ...", flush=True)
    for b in range(B):
        ib = block_indices(len(yte), block, rng)
        io = block_indices(len(yo), block, rng)
        boot["benchmark"]["peak"][b] = peak_tss(yte[ib], pte[ib])
        boot["benchmark"]["frozen"][b] = frozen_tss(yte[ib], pte[ib], thr)
        boot["operational"]["peak"][b] = peak_tss(yo[io], po[io])
        boot["operational"]["frozen"][b] = frozen_tss(yo[io], po[io], thr)
    gap_peak_boot = boot["benchmark"]["peak"] - boot["operational"]["peak"]
    gap_frozen_boot = boot["benchmark"]["frozen"] - boot["operational"]["frozen"]

    b_peak, o_peak = peak_tss(yte, pte), peak_tss(yo, po)
    b_frozen, o_frozen = frozen_tss(yte, pte, thr), frozen_tss(yo, po, thr)
    out = {
        "title": "Storm model: benchmark vs. operational skill (generalisation check)",
        "task": f"P(Kp>={cfg['storm']['kp_storm_threshold']:g} within "
                f"{cfg['storm']['prediction_window_h']}h) from L1 solar wind (OMNI)",
        "winner": winner,
        "eras": {
            "benchmark": f"{bench_span[0]}..{bench_span[1]} (held-out chronological test: "
                         f"{int(len(yte))} windows, {int(yte.sum())} storms)",
            "operational": f"{op_span[0]}..{op_span[1]} (unseen later era, different "
                           f"solar-cycle phase: {int(len(yo))} windows, {int(yo.sum())} storms)",
        },
        "base_rates": {"benchmark_test": round(float(yte.mean()), 4),
                       "operational": round(float(yo.mean()), 4)},
        "peak": {
            "benchmark_tss": round(b_peak, 3), "benchmark_ci": _ci(boot["benchmark"]["peak"], level),
            "operational_tss": round(o_peak, 3), "operational_ci": _ci(boot["operational"]["peak"], level),
            "gap": round(b_peak - o_peak, 3), "gap_ci": _ci(gap_peak_boot, level),
        },
        "frozen": {
            "threshold": round(float(thr), 3),
            "benchmark_tss": round(b_frozen, 3), "benchmark_ci": _ci(boot["benchmark"]["frozen"], level),
            "operational_tss": round(o_frozen, 3), "operational_ci": _ci(boot["operational"]["frozen"], level),
            "gap": round(b_frozen - o_frozen, 3), "gap_ci": _ci(gap_frozen_boot, level),
        },
        "method": {
            "bootstrap": f"moving-block bootstrap, {B} replicates, "
                         f"{int(sc.get('block_days', 7))}-day blocks (storm windows "
                         "autocorrelate — i.i.d. resampling would understate uncertainty)",
            "note": "operational era spans a different solar-cycle phase; that IS part "
                    "of what deployment faces, not a confound to remove",
        },
        "findings": {
            "benchmarks_overstate": bool(np.percentile(gap_peak_boot, (1 - level) / 2 * 100) > 0),
            "overstatement_gap": round(b_peak - o_peak, 3),
            "overstatement_gap_ci": _ci(gap_peak_boot, level),
            "frozen_overstatement_gap": round(b_frozen - o_frozen, 3),
            "frozen_overstatement_gap_ci": _ci(gap_frozen_boot, level),
        },
    }
    path = os.path.join(cfg["_project_root"], "storm_scorecard.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {path}", flush=True)
    return out


def main():
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
