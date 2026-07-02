"""The project's research experiment, made reproducible AND statistically robust.

Research question: *Do standard benchmark scores overstate the real-time
operational skill of ML flare forecasts, and does training on live satellite data
close the gap?*

A 2x2: two training sources x two kinds of test set, scored with TSS.

               |  BENCHMARK test (held-out SWAN-SF)  |  OPERATIONAL test (live JSOC, unseen years)
  ------------ | ----------------------------------- | ------------------------------------------
  Benchmark-   |          (a)                        |            (b)   <- overstatement = a - b
   trained     |                                     |
  Live-trained |          (c)                        |            (d)

  * Benchmarks overstate operational skill  <=>  a - b > 0 (and its 95% CI excludes 0)
  * Live training closes the gap            <=>  (c - d) < (a - b)

Statistical robustness (v2):
  * MULTIPLE operational years — every data/sharp_live/dataset_YYYY.npz that is
    unseen by BOTH models (2013/2015/2016/2017 when built) is scored, so the gap
    is shown to hold across years, not once.
  * BOOTSTRAP 95% CIs — test sets are resampled (paired across models, so the
    model comparison shares the same resampled rows) and every TSS and every gap
    gets an uncertainty interval.
  * FROZEN-THRESHOLD scores — peak TSS lets each test set pick its best threshold
    after the fact (a fair ceiling for cross-set comparison, but generous).
    Real operations must commit to a threshold in advance, so we ALSO score each
    model with its validation-tuned threshold frozen. That is the deployment
    number, and the gap there is the most honest version of the finding.
  * RELIABILITY data — binned predicted-vs-observed frequencies on the pooled
    operational years (+ Brier score), for the calibration diagram.

Both models are trained in-memory (save=False) so the deployed model is untouched.
Writes skill_scorecard.json for the dashboard "Model Skill Scorecard" panel.

Run:  python -m solarflare.scorecard
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime

import numpy as np

from . import data as dataio
from . import evaluate as ev
from . import sharpdata
from . import sharptrain
from .config import load_config

# Years that are training-era for one of the two models and therefore can never
# serve as a neutral operational test: 2010-2012 = SWAN-SF partition-1 span
# (benchmark model), 2014 = the live model's training year.
_TRAINING_ERA_YEARS = {2010, 2011, 2012, 2014}


# ----------------------------------------------------------------------
# Fast, exact skill measures (vectorised so the bootstrap is cheap)
# ----------------------------------------------------------------------
def peak_tss(y, p) -> float:
    """Best-achievable TSS over ALL thresholds (Youden's J via the ROC curve).
    Threshold-free skill ceiling; O(n log n) so 1000s of bootstrap replicates
    stay fast. Never below 0 (the 'never warn' strategy)."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys, dtype=np.int64)
    fp = np.cumsum(1 - ys, dtype=np.int64)
    npos, nneg = int(tp[-1]), int(fp[-1])
    if npos == 0 or nneg == 0:
        return 0.0
    ok = np.append(ps[1:] != ps[:-1], True)      # thresholds only between distinct probs
    curve = tp[ok] / npos - fp[ok] / nneg
    return float(max(0.0, curve.max()))


def frozen_tss(y, p, thr: float) -> float:
    """TSS with a PRE-COMMITTED threshold — the honest deployment score."""
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(p, dtype=float) >= thr
    npos = int(y.sum())
    nneg = int(len(y) - npos)
    if npos == 0 or nneg == 0:
        return 0.0
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    return tp / npos - fp / nneg


# ----------------------------------------------------------------------
# Bootstrap machinery
# ----------------------------------------------------------------------
def _ci(samples, level: float):
    lo = float(np.percentile(samples, (1 - level) / 2 * 100))
    hi = float(np.percentile(samples, (1 + level) / 2 * 100))
    return [round(lo, 3), round(hi, 3)]


def bootstrap_sets(y_by_set: dict, p_by_set_model: dict, thr_by_model: dict,
                   groups_by_set: dict, B: int, rng,
                   thr2_by_model: dict | None = None) -> dict:
    """CLUSTER bootstrap by active region: windows of the same region overlap in
    time and are strongly correlated, so the honest resampling unit is the whole
    region, not the row (i.i.d. rows would give dishonestly narrow CIs).

    Within one replicate the SAME resampled regions are scored by every model
    (paired — the fair way to compare models); different sets resample
    independently (they are independent samples).

    `thr2_by_model` optionally supplies a SECOND frozen threshold per model
    (e.g. the operationally-recalibrated one); its scores land in "frozen2" on
    the same replicates, so frozen2 − frozen is a PAIRED recovery estimate.

    Returns  boot[set][model] = {"peak": ndarray(B), "frozen": ndarray(B)
                                 [, "frozen2": ndarray(B)]}."""
    thr2_by_model = thr2_by_model or {}
    boot = {s: {m: ({"peak": np.empty(B), "frozen": np.empty(B), "frozen2": np.empty(B)}
                    if m in thr2_by_model else
                    {"peak": np.empty(B), "frozen": np.empty(B)})
                for m in thr_by_model} for s in y_by_set}
    for s, y in y_by_set.items():
        g = np.asarray(groups_by_set[s])
        clusters = [np.where(g == u)[0] for u in np.unique(g)]
        k = len(clusters)
        for b in range(B):
            pick = rng.integers(0, k, k)
            idx = np.concatenate([clusters[i] for i in pick])
            yb = y[idx]
            for m in thr_by_model:
                pb = p_by_set_model[s][m][idx]
                boot[s][m]["peak"][b] = peak_tss(yb, pb)
                boot[s][m]["frozen"][b] = frozen_tss(yb, pb, thr_by_model[m])
                if m in thr2_by_model:
                    boot[s][m]["frozen2"][b] = frozen_tss(yb, pb, thr2_by_model[m])
    return boot


# ----------------------------------------------------------------------
# Reliability (calibration) data
# ----------------------------------------------------------------------
def reliability_bins(y, p, n_bins: int = 10) -> dict:
    """Fixed-width probability bins -> observed flare frequency per bin, plus the
    Brier score. This is the 'does 30% mean 30%?' check, on operational data."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({"p_lo": round(float(edges[i]), 2), "p_hi": round(float(edges[i + 1]), 2),
                     "p_mean": round(float(p[m].mean()), 4),
                     "obs_freq": round(float(y[m].mean()), 4), "n": n})
    return {"bins": rows,
            "brier": round(float(np.mean((p - y) ** 2)), 5),
            "base_rate": round(float(y.mean()), 5)}


# ----------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------
def detect_operational_years(data_dir: str) -> list[int]:
    """Every dataset_YYYY.npz on disk that is unseen by BOTH models."""
    years = []
    for fn in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
        m = re.fullmatch(r"dataset_(\d{4})\.npz", fn)
        if m and int(m.group(1)) not in _TRAINING_ERA_YEARS:
            years.append(int(m.group(1)))
    return sorted(years)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    root = cfg["_project_root"]
    dd = os.path.join(root, "data", "sharp_live")
    sc = cfg.get("scorecard", {}) or {}
    B = int(sc.get("bootstrap_replicates", 1000))
    level = float(sc.get("ci_level", 0.95))
    n_bins = int(sc.get("reliability_bins", 10))
    rng = np.random.default_rng(int(sc.get("random_state", 42)))

    # Two models, trained in-memory (deployed model untouched):
    print("Training benchmark model (SWAN-SF 2010-2012) ...", flush=True)
    v2 = sharptrain.train(os.path.join(dd, "dataset_swansf_p1.npz"), cfg, save=False)
    print("Training live model (JSOC 2014) ...", flush=True)
    v1 = sharptrain.train(os.path.join(dd, "dataset_2014.npz"), cfg, save=False)
    models = {"Benchmark-trained": v2, "Live-trained": v1}
    details = {"Benchmark-trained": "SWAN-SF 2010-2012", "Live-trained": "JSOC live data (2014)"}

    # Benchmark test split, loaded EARLY: the multi-year model's live 2011/2012
    # training data overlaps SWAN-SF p1 in both time and region ids (SWAN-SF's
    # ar<N> = HARPNUM), so its regions must be excluded from that model's
    # training set or the benchmark column would be scored on training regions.
    dp = sharpdata.load_dataset(os.path.join(dd, "dataset_swansf_p1.npz"))
    _, _, ite = sharptrain.time_group_split(dp["groups"], dp["end_times"],
                                            cfg["sharp_live"]["split"])
    bench_test_regions = np.unique(dp["groups"][ite])

    # Optional third model — does MORE live data close the gap? Trains on every
    # pre-2015 live year on disk; scored only on strictly-later years.
    MULTI = "Live-trained (multi-year)"
    multi_train_years = [y for y in (2011, 2012, 2014)
                         if os.path.exists(os.path.join(dd, f"dataset_{y}.npz"))]
    multi_dropped = 0
    if bool(sc.get("include_multiyear_live", True)) and len(multi_train_years) >= 2:
        print(f"Training live model (JSOC {'+'.join(map(str, multi_train_years))}) ...",
              flush=True)
        parts = []
        for y in multi_train_years:
            p = sharpdata.load_dataset(os.path.join(dd, f"dataset_{y}.npz"))
            keep = ~np.isin(p["groups"], bench_test_regions)   # leakage guard
            multi_dropped += int((~keep).sum())
            parts.append({"X3d": p["X3d"][keep], "y": p["y"][keep],
                          "groups": p["groups"][keep],
                          "end_times": [t for t, k in zip(p["end_times"], keep) if k]})
        if multi_dropped:
            print(f"  leakage guard: dropped {multi_dropped} training windows from "
                  f"regions in the benchmark test split", flush=True)
        d_multi = {"X3d": np.concatenate([p["X3d"] for p in parts]),
                   "y": np.concatenate([p["y"] for p in parts]),
                   "groups": np.concatenate([p["groups"] for p in parts]),
                   "end_times": [t for p in parts for t in p["end_times"]]}
        models[MULTI] = sharptrain.train(d_multi, cfg, save=False)
        details[MULTI] = "JSOC " + "+".join(str(y) for y in multi_train_years)
    # Deployment thresholds: the validation-TSS-optimal threshold each model
    # committed to BEFORE seeing any test set (operating_points.high_recall is
    # argmax-TSS-on-val — see evaluate.operating_points).
    thr = {m: float(v["operating_points"]["high_recall"]["threshold"])
           for m, v in models.items()}

    # BENCHMARK test = held-out region-disjoint split of SWAN-SF p1 (no model
    # trained on those regions — see the leakage guard above). OPERATIONAL
    # tests = every unseen JSOC year.
    y_by_set = {"benchmark": dp["y"][ite]}
    X_by_set = {"benchmark": dataio.build_matrix(dp["X3d"][ite], cfg)}
    groups_by_set = {"benchmark": dp["groups"][ite]}
    years = detect_operational_years(dd)
    if not years:
        raise RuntimeError("no operational dataset_YYYY.npz found in data/sharp_live")
    for yr in years:
        do = sharpdata.load_dataset(os.path.join(dd, f"dataset_{yr}.npz"))
        y_by_set[str(yr)] = do["y"]
        X_by_set[str(yr)] = dataio.build_matrix(do["X3d"], cfg)
        groups_by_set[str(yr)] = do["groups"]
    print(f"Operational years on disk: {years}", flush=True)

    # Score every model on every set (probabilities computed once, reused by the bootstrap).
    p_by_set_model = {s: {m: v["model"].predict_proba(X_by_set[s])[:, 1]
                          for m, v in models.items()} for s in y_by_set}

    op_sets = [str(yr) for yr in years]

    def _valid_sets(model_name: str) -> list[str]:
        """The multi-year model may only be scored on years strictly AFTER its
        training span; the other two models can use every detected year."""
        if model_name == MULTI:
            cut = max(multi_train_years) if multi_train_years else 2014
            return [s for s in op_sets if int(s) > cut]
        return op_sets

    # Self-correction experiment (mirrors `python -m solarflare.recalibrate`,
    # in-memory only): recalibrate each model's threshold on its EARLIEST
    # operational year strictly after its own training span, then freeze that
    # threshold for the strictly-later years.
    def _cutoff_year(payload) -> int:
        try:
            return datetime.fromisoformat(str(payload["data_span"][1]).replace("Z", "+00:00")).year
        except (KeyError, IndexError, ValueError, TypeError):
            return 2014
    recal = {}
    for m, v in models.items():
        after = [s for s in _valid_sets(m) if int(s) > _cutoff_year(v)]
        if len(after) < 2:
            continue                                  # need a cal year AND eval years
        cal = after[0]
        thr_r, _ = ev.best_threshold(y_by_set[cal], p_by_set_model[cal][m], "tss")
        recal[m] = {"cal_year": int(cal), "threshold": float(thr_r),
                    "eval_years": after[1:]}
        print(f"  recalibrated {m}: thr {thr[m]:.3f} -> {thr_r:.3f} "
              f"(on {cal}; eval {after[1:]})", flush=True)

    print(f"Bootstrapping ({B} replicates, {int(level*100)}% CI, cluster-by-region) ...",
          flush=True)
    boot = bootstrap_sets(y_by_set, p_by_set_model, thr, groups_by_set, B, rng,
                          thr2_by_model={m: r["threshold"] for m, r in recal.items()})

    rows = []
    for m, v in models.items():
        sets_m = _valid_sets(m)
        if not sets_m:
            continue                                   # no usable operational year yet
        yb, pb = y_by_set["benchmark"], p_by_set_model["benchmark"][m]
        b_peak = peak_tss(yb, pb)
        b_frozen = frozen_tss(yb, pb, thr[m])
        by_year = {}
        for s in sets_m:
            ys_, ps_ = y_by_set[s], p_by_set_model[s][m]
            by_year[s] = {
                "peak_tss": round(peak_tss(ys_, ps_), 3),
                "peak_ci": _ci(boot[s][m]["peak"], level),
                "frozen_tss": round(frozen_tss(ys_, ps_, thr[m]), 3),
                "frozen_ci": _ci(boot[s][m]["frozen"], level),
                "n": int(len(ys_)), "positives": int(ys_.sum()),
            }
        # Operational skill = mean across years; its bootstrap = mean of the
        # per-year replicates (sets resampled independently within a replicate).
        op_peak_boot = np.mean([boot[s][m]["peak"] for s in sets_m], axis=0)
        op_frozen_boot = np.mean([boot[s][m]["frozen"] for s in sets_m], axis=0)
        op_peak = float(np.mean([by_year[s]["peak_tss"] for s in sets_m]))
        op_frozen = float(np.mean([by_year[s]["frozen_tss"] for s in sets_m]))
        gap_boot = boot["benchmark"][m]["peak"] - op_peak_boot
        gap_frozen_boot = boot["benchmark"][m]["frozen"] - op_frozen_boot

        # Self-corrected deployment: the recalibrated threshold frozen on the
        # strictly-later years, with a PAIRED recovery CI (same replicates).
        recal_block = None
        r_info = recal.get(m)
        if r_info:
            ev_yrs = r_info["eval_years"]
            by_year_recal = {
                s: {"frozen_tss": round(frozen_tss(y_by_set[s], p_by_set_model[s][m],
                                                   r_info["threshold"]), 3),
                    "frozen_ci": _ci(boot[s][m]["frozen2"], level)}
                for s in ev_yrs}
            recal_op_boot = np.mean([boot[s][m]["frozen2"] for s in ev_yrs], axis=0)
            before_boot = np.mean([boot[s][m]["frozen"] for s in ev_yrs], axis=0)
            recal_op = float(np.mean([by_year_recal[s]["frozen_tss"] for s in ev_yrs]))
            before = float(np.mean([frozen_tss(y_by_set[s], p_by_set_model[s][m], thr[m])
                                    for s in ev_yrs]))
            recal_block = {
                "calibrated_on": r_info["cal_year"],
                "threshold": round(r_info["threshold"], 3),
                "eval_years": [int(s) for s in ev_yrs],
                "operational_tss": round(recal_op, 3),
                "operational_ci": _ci(recal_op_boot, level),
                "before_tss_same_years": round(before, 3),
                "recovery": round(recal_op - before, 3),
                "recovery_ci": _ci(recal_op_boot - before_boot, level),
                # Before/after gaps on the SAME eval years (the headline gap
                # averages more/earlier years — juxtaposing it against gap_after
                # would conflate the year set change with the recalibration).
                "gap_before_same_years": round(b_frozen - before, 3),
                "gap_before_same_years_ci": _ci(boot["benchmark"][m]["frozen"] - before_boot, level),
                "gap_after": round(b_frozen - recal_op, 3),
                "gap_after_ci": _ci(boot["benchmark"][m]["frozen"] - recal_op_boot, level),
                "by_year": by_year_recal,
            }

        rows.append({
            "name": m, "detail": details[m],
            "operational_years_used": [int(s) for s in sets_m],
            # Backward-compatible headline keys (peak TSS, mean over years):
            "benchmark_tss": round(b_peak, 3),
            "operational_tss": round(op_peak, 3),
            "gap": round(b_peak - op_peak, 3),
            # v2 additions:
            "benchmark_ci": _ci(boot["benchmark"][m]["peak"], level),
            "operational_ci": _ci(op_peak_boot, level),
            "gap_ci": _ci(gap_boot, level),
            "frozen": {
                "threshold": round(thr[m], 3),
                "benchmark_tss": round(b_frozen, 3),
                "benchmark_ci": _ci(boot["benchmark"][m]["frozen"], level),
                "operational_tss": round(op_frozen, 3),
                "operational_ci": _ci(op_frozen_boot, level),
                "gap": round(b_frozen - op_frozen, 3),
                "gap_ci": _ci(gap_frozen_boot, level),
            },
            "recalibrated": recal_block,
            "by_year": by_year,
            "_gap_boot": gap_boot,                # internal, stripped before writing
            "_gap_frozen_boot": gap_frozen_boot,
        })

    bench = next(r for r in rows if r["name"] == "Benchmark-trained")
    live = next(r for r in rows if r["name"] == "Live-trained")
    # Does live training close the gap? CI on (benchmark model's gap - live
    # model's gap): > 0 means live training genuinely shrinks the gap.
    closes_boot = bench["_gap_boot"] - live["_gap_boot"]
    closes_ci = _ci(closes_boot, level)

    # Does MORE live data close the gap? Compare single-year vs multi-year live
    # models on the SAME (strictly post-training) years — paired via the shared
    # bootstrap replicates.
    more_data = None
    multi = next((r for r in rows if r["name"] == MULTI), None)
    if multi is not None:
        yrs = [str(y) for y in multi["operational_years_used"]]
        live_op_same = np.mean([boot[s]["Live-trained"]["peak"] for s in yrs], axis=0)
        live_gap_same_boot = boot["benchmark"]["Live-trained"]["peak"] - live_op_same
        diff_boot = live_gap_same_boot - multi["_gap_boot"]   # >0 => multi-year gap smaller
        live_gap_same = round(live["benchmark_tss"] - float(
            np.mean([live["by_year"][s]["peak_tss"] for s in yrs])), 3)
        more_data = {
            "years_compared": [int(s) for s in yrs],
            "single_year_gap_same_years": live_gap_same,
            "multi_year_gap": multi["gap"],
            "more_data_closes_gap": bool(multi["gap"] < live_gap_same),
            "single_minus_multi_gap_ci": _ci(diff_boot, level),
        }
    # The self-correction story: does recalibrating the threshold on ONE
    # operational year recover the lost skill — and does combining that with
    # multi-year live training close the gap entirely?
    recal_findings = None
    if bench.get("recalibrated"):
        rb = bench["recalibrated"]
        recal_findings = {
            "benchmark_trained": {
                "recovery": rb["recovery"], "recovery_ci": rb["recovery_ci"],
                "gap_before_same_years": rb["gap_before_same_years"],
                "gap_after": rb["gap_after"], "gap_after_ci": rb["gap_after_ci"],
                "recovers_significantly": bool(rb["recovery_ci"][0] > 0),
            },
        }
        if multi is not None and multi.get("recalibrated"):
            mb = multi["recalibrated"]
            recal_findings["combined_fix"] = {
                "description": "multi-year live training + one-year threshold recalibration",
                "operational_tss": mb["operational_tss"],
                "operational_ci": mb["operational_ci"],
                "recovery": mb["recovery"], "recovery_ci": mb["recovery_ci"],
                "gap_before_same_years": mb["gap_before_same_years"],
                "gap_after": mb["gap_after"], "gap_after_ci": mb["gap_after_ci"],
                # "Closed" requires the POINT estimate at/below zero — a CI that
                # merely spans zero is absence of evidence, not evidence of
                # absence, and is reported separately as not-significant.
                "gap_closed": bool(mb["gap_after"] <= 0),
                "gap_not_significant": bool(mb["gap_after_ci"][0] <= 0 <= mb["gap_after_ci"][1]),
            }

    for r in rows:
        r.pop("_gap_boot"), r.pop("_gap_frozen_boot")

    # Reliability on the pooled operational years (each model only on the years
    # it may legitimately be scored on).
    reliability = {}
    for m in models:
        sets_m = _valid_sets(m)
        if not sets_m:
            continue
        y_pool = np.concatenate([y_by_set[s] for s in sets_m])
        p_pool = np.concatenate([p_by_set_model[s][m] for s in sets_m])
        reliability[m] = reliability_bins(y_pool, p_pool, n_bins)

    out = {
        "title": "Benchmark score vs. real operational skill",
        "metric": "TSS (True Skill Statistic; higher is better, 0 = no skill)",
        "method": {
            "peak": "best TSS over all thresholds (Youden's J) — a threshold-free "
                    "skill ceiling, fair across test sets but generous",
            "frozen": "TSS with the validation-tuned threshold committed in advance "
                      "— the honest deployment score",
            "bootstrap": f"{B} cluster-bootstrap resamples (whole active regions, "
                         "because windows of one region are correlated), "
                         f"{int(level*100)}% percentile CIs, paired across models",
            "climatology_note": "a constant-probability (climatology) forecast has "
                                "TSS = 0 by construction — the zero line IS the "
                                "climatology baseline",
            "leakage_guard": f"multi-year live training excluded {int(multi_dropped)} "
                             "windows from regions present in the benchmark test split "
                             "(live 2011/2012 JSOC data overlaps SWAN-SF p1 in time and "
                             "region ids)",
        },
        "test_sets": {
            "benchmark": f"held-out SWAN-SF magnetograms ({int(len(y_by_set['benchmark']))} windows, "
                         f"{int(y_by_set['benchmark'].sum())} flares)",
            "operational": "live JSOC, unseen year(s): " + ", ".join(
                f"{s} ({int(len(y_by_set[s]))} windows, {int(y_by_set[s].sum())} flares)"
                for s in op_sets),
            "operational_years": [int(s) for s in op_sets],
        },
        "models": rows,
        "reliability": reliability,
        "findings": {
            "benchmarks_overstate": bool(bench["gap_ci"][0] > 0),
            "overstatement_gap": bench["gap"],
            "overstatement_gap_ci": bench["gap_ci"],
            "frozen_overstatement_gap": bench["frozen"]["gap"],
            "frozen_overstatement_gap_ci": bench["frozen"]["gap_ci"],
            "gap_positive_every_year": bool(all(
                bench["benchmark_tss"] - v["peak_tss"] > 0 for v in bench["by_year"].values())),
            "live_training_closes_gap": bool(live["gap"] < bench["gap"]),
            "live_minus_benchmark_gap_ci": closes_ci,
            "more_live_data": more_data,
            "recalibration": recal_findings,
        },
    }
    path = os.path.join(root, "skill_scorecard.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {path}", flush=True)
    return out


def main():
    out = run()
    print(json.dumps({k: v for k, v in out.items() if k != "reliability"}, indent=2))


if __name__ == "__main__":
    main()
