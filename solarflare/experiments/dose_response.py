"""Dose-response: does the benchmark-operational gap shrink as the amount of
LIVE training data grows?

The scorecard's single data point (multi-year training narrows the gap, n.s.)
begs the real question: what is the TREND? This experiment trains the same
pipeline on accumulating live year-sets —

    [2014]  ->  [2013-14]  ->  [2012-14]  ->  [2011-14]

— and scores every model on the SAME fixed test sets: the SWAN-SF benchmark
held-out split and every operational year strictly after 2014 (2015/16/17,
plus 2023 when built). All training steps end in 2014, so the operational test
years stay untouched for every step. The BENCHMARK split needs one extra guard:
live 2011/2012 JSOC data overlaps SWAN-SF p1 in time and region ids, so
training windows from regions in the benchmark test split are dropped —
otherwise the larger steps would be partially scored on their own training
regions, inflating exactly the trend this experiment measures.

Peak + frozen TSS with cluster-by-region bootstrap CIs, exactly like the
scorecard. Trains in-memory only (deployed models untouched).

Writes dose_response.json (merged into /api/scorecard by the server; drawn as
the "gap vs. training years" chart in the dashboard scorecard panel).

Run:  python -m solarflare.experiments.dose_response
"""
from __future__ import annotations

import json
import os

import numpy as np

from .. import data as dataio
from .. import sharpdata, sharptrain
from ..config import load_config
from ..scorecard import (_ci, bootstrap_sets, detect_operational_years,
                         frozen_tss, peak_tss)

# Accumulate BACKWARDS from 2014 so every step ends at the same date and the
# post-2014 test years remain valid for all of them.
STEPS = [[2014], [2013, 2014], [2012, 2013, 2014], [2011, 2012, 2013, 2014]]


def _load_concat(dd: str, years: list[int], exclude_regions=None):
    """Concatenate year datasets; `exclude_regions` drops every window whose
    region sits in the benchmark TEST split — live 2011/2012 JSOC data overlaps
    SWAN-SF p1 in time and region ids (ar<N> = HARPNUM), so without this guard
    the benchmark score of the larger training steps would be measured on
    training regions, inflating exactly the trend this experiment exists to
    measure. Returns (dataset, n_dropped)."""
    parts, dropped = [], 0
    for y in years:
        p = sharpdata.load_dataset(os.path.join(dd, f"dataset_{y}.npz"))
        if exclude_regions is not None:
            keep = ~np.isin(p["groups"], exclude_regions)
            dropped += int((~keep).sum())
            p = {"X3d": p["X3d"][keep], "y": p["y"][keep],
                 "groups": p["groups"][keep],
                 "end_times": [t for t, k in zip(p["end_times"], keep) if k]}
        parts.append(p)
    return {"X3d": np.concatenate([p["X3d"] for p in parts]),
            "y": np.concatenate([p["y"] for p in parts]),
            "groups": np.concatenate([p["groups"] for p in parts]),
            "end_times": [t for p in parts for t in p["end_times"]]}, dropped


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    root = cfg["_project_root"]
    dd = os.path.join(root, "data", "sharp_live")
    dr = cfg.get("dose_response", {}) or {}
    B = int(dr.get("bootstrap_replicates", 500))
    level = float(dr.get("ci_level", 0.95))
    rng = np.random.default_rng(int(dr.get("random_state", 42)))

    steps = [s for s in STEPS
             if all(os.path.exists(os.path.join(dd, f"dataset_{y}.npz")) for y in s)]
    if not steps:
        raise RuntimeError("no complete training year-set on disk — build dataset_2014.npz first")
    test_years = [y for y in detect_operational_years(dd) if y > 2014]
    if not test_years:
        raise RuntimeError("no operational test year after 2014 on disk")
    print(f"Steps: {steps}\nFixed test years: {test_years}", flush=True)

    # Fixed test sets, built once (benchmark = held-out SWAN-SF split).
    dp = sharpdata.load_dataset(os.path.join(dd, "dataset_swansf_p1.npz"))
    _, _, ite = sharptrain.time_group_split(dp["groups"], dp["end_times"],
                                            cfg["sharp_live"]["split"])
    y_by_set = {"benchmark": dp["y"][ite]}
    X_by_set = {"benchmark": dataio.build_matrix(dp["X3d"][ite], cfg)}
    groups_by_set = {"benchmark": dp["groups"][ite]}
    for yr in test_years:
        d = sharpdata.load_dataset(os.path.join(dd, f"dataset_{yr}.npz"))
        y_by_set[str(yr)] = d["y"]
        X_by_set[str(yr)] = dataio.build_matrix(d["X3d"], cfg)
        groups_by_set[str(yr)] = d["groups"]
    op_sets = [str(y) for y in test_years]

    bench_test_regions = np.unique(dp["groups"][ite])

    out_steps = []
    for years in steps:
        name = "+".join(map(str, years))
        print(f"\n=== training on {name} ===", flush=True)
        d, dropped = _load_concat(dd, years, exclude_regions=bench_test_regions)
        if dropped:
            print(f"  leakage guard: dropped {dropped} windows from regions in the "
                  f"benchmark test split", flush=True)
        payload = sharptrain.train(d, cfg, save=False)
        thr = float(payload["operating_points"]["high_recall"]["threshold"])
        p_by = {s: {name: payload["model"].predict_proba(X_by_set[s])[:, 1]}
                for s in y_by_set}
        print(f"  bootstrapping ({B} replicates) ...", flush=True)
        boot = bootstrap_sets(y_by_set, p_by, {name: thr}, groups_by_set, B, rng)

        b_peak = peak_tss(y_by_set["benchmark"], p_by["benchmark"][name])
        b_frozen = frozen_tss(y_by_set["benchmark"], p_by["benchmark"][name], thr)
        by_year = {s: round(peak_tss(y_by_set[s], p_by[s][name]), 3) for s in op_sets}
        op_peak = float(np.mean(list(by_year.values())))
        op_frozen = float(np.mean([frozen_tss(y_by_set[s], p_by[s][name], thr)
                                   for s in op_sets]))
        op_peak_boot = np.mean([boot[s][name]["peak"] for s in op_sets], axis=0)
        op_frozen_boot = np.mean([boot[s][name]["frozen"] for s in op_sets], axis=0)
        out_steps.append({
            "train_years": years,
            "n_train": int(len(d["y"])),
            "train_positives": int(d["y"].sum()),
            "train_windows_dropped_leakage_guard": int(dropped),
            "benchmark_tss": round(b_peak, 3),
            "benchmark_ci": _ci(boot["benchmark"][name]["peak"], level),
            "operational_tss": round(op_peak, 3),
            "operational_ci": _ci(op_peak_boot, level),
            "gap": round(b_peak - op_peak, 3),
            "gap_ci": _ci(boot["benchmark"][name]["peak"] - op_peak_boot, level),
            "frozen_gap": round(b_frozen - op_frozen, 3),
            "frozen_gap_ci": _ci(boot["benchmark"][name]["frozen"] - op_frozen_boot, level),
            "operational_by_year": by_year,
        })
        print(f"  bench {b_peak:.3f}  op {op_peak:.3f}  gap {b_peak-op_peak:+.3f}", flush=True)

    out = {
        "title": "Dose-response: gap vs. amount of live training data",
        "test_years": test_years,
        "metric": "peak TSS (cluster-bootstrap 95% CIs); frozen gap uses the "
                  "validation-tuned threshold committed in advance",
        "method": "same pipeline, accumulating live training year-sets all ending "
                  "2014; identical fixed test sets for every step, so the points "
                  "are directly comparable. Leakage guard: training windows from "
                  "regions present in the benchmark test split are excluded (live "
                  "2011/2012 JSOC data overlaps SWAN-SF p1 in time and region ids).",
        "steps": out_steps,
    }
    path = os.path.join(root, "dose_response.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {path}", flush=True)
    return out


def main():
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
