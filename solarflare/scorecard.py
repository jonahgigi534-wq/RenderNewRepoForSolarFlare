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

Protocol hardening (v3):
  * NO BACKWARDS TESTING — every live-trained model is scored only on years
    strictly after its training data (a deployed system cannot be tested on its
    past), and cross-model gap comparisons use identical year sets.
  * BENCHMARK LEAKAGE GUARD — a model whose live training years overlap the
    SWAN-SF test split's era (the multi-year model: JSOC 2011+2012 vs a test
    split spanning late-2011..2012-03) gets no benchmark/gap cells: the same
    active regions would sit in both training and test, and no leakage-free
    benchmark subset exists. It is scored on operational years only.

Both models are trained in-memory (save=False) so the deployed model is untouched.
Writes skill_scorecard.json for the dashboard "Model Skill Scorecard" panel.

Run:  python -m solarflare.scorecard
"""
from __future__ import annotations

import json
import os
import re

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
def label_gate_status(cfg: dict | None, year: int) -> tuple[bool, str]:
    """FAIL-CLOSED label-quality check for one year. Our labeler can only mark a
    positive when HEK/SWPC gives the flare a NOAA AR number, so a year needs a
    MEASURED attribution rate at or above scorecard.min_label_attribution to be
    scored — an unmeasured year is excluded too (silence must not pass the
    gate; 2023 taught us why). Returns (ok, reason_if_not)."""
    sc = (cfg or {}).get("scorecard", {}) or {}
    gate = float(sc.get("min_label_attribution", 0.0))
    if gate <= 0:
        return True, ""
    rates = {int(y): float(r)
             for y, r in (sc.get("label_attribution_by_year") or {}).items()}
    r = rates.get(int(year))
    if r is None:
        return False, (f"label-attribution rate unmeasured (gate {gate:g}) — run "
                       f"scripts/label_attribution.py {year} {year} and record it "
                       "in config scorecard.label_attribution_by_year")
    if r < gate:
        return False, (f"label-attribution rate {r:.2f} below the {gate:g} gate "
                       "(HEK/SWPC AR attribution incomplete; labels under-count "
                       "positives, biasing TSS down)")
    return True, ""


def label_excluded_years(cfg: dict | None) -> dict[int, float]:
    """Years whose MEASURED attribution rate is below the gate, {year: rate}.
    (Unmeasured years are additionally excluded by label_gate_status /
    detect_operational_years — this map only lists the measured-bad ones,
    for artifact annotations.)"""
    sc = (cfg or {}).get("scorecard", {}) or {}
    gate = float(sc.get("min_label_attribution", 0.0))
    rates = sc.get("label_attribution_by_year", {}) or {}
    return {int(y): float(r) for y, r in rates.items() if float(r) < gate}


def detect_operational_years(data_dir: str, cfg: dict | None = None) -> list[int]:
    """Every dataset_YYYY.npz on disk that is unseen by BOTH models. Pass cfg to
    also apply the label-attribution gate (required for label-dependent
    evaluations; noaa_baseline legitimately omits it). The gate is fail-closed:
    a year without a measured attribution rate is excluded until measured."""
    years = []
    for fn in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
        m = re.fullmatch(r"dataset_(\d{4})\.npz", fn)
        if not m:
            continue
        yr = int(m.group(1))
        if yr in _TRAINING_ERA_YEARS:
            continue
        if cfg is not None and not label_gate_status(cfg, yr)[0]:
            continue
        years.append(yr)
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

    # Optional third model — does MORE live data close the gap? Trains on every
    # pre-2015 live year on disk; scored only on strictly-later years.
    MULTI = "Live-trained (multi-year)"
    multi_train_years = [y for y in (2011, 2012, 2014)
                         if os.path.exists(os.path.join(dd, f"dataset_{y}.npz"))]
    if bool(sc.get("include_multiyear_live", True)) and len(multi_train_years) >= 2:
        print(f"Training live model (JSOC {'+'.join(map(str, multi_train_years))}) ...",
              flush=True)
        parts = [sharpdata.load_dataset(os.path.join(dd, f"dataset_{y}.npz"))
                 for y in multi_train_years]
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

    # BENCHMARK test = held-out region-disjoint split of SWAN-SF p1 (neither model
    # trained on those exact windows). OPERATIONAL tests = every unseen JSOC year.
    dp = sharpdata.load_dataset(os.path.join(dd, "dataset_swansf_p1.npz"))
    _, _, ite = sharptrain.time_group_split(dp["groups"], dp["end_times"],
                                            cfg["sharp_live"]["split"])
    y_by_set = {"benchmark": dp["y"][ite]}
    X_by_set = {"benchmark": dataio.build_matrix(dp["X3d"][ite], cfg)}
    groups_by_set = {"benchmark": dp["groups"][ite]}

    # Leakage guard: the multi-year model trains on live JSOC years that overlap
    # the SWAN-SF test split's era — same Sun, so the same active regions appear
    # in its training data and the benchmark test. When the eras overlap, NO
    # leakage-free benchmark cell exists for that model (the test split's whole
    # time span falls inside the training era) and it is scored on operational
    # years only.
    bench_test_years = {int(str(dp["end_times"][i])[:4]) for i in ite}
    multi_bench_leaky = MULTI in models and bool(
        set(multi_train_years) & bench_test_years)
    excluded = label_excluded_years(cfg)
    if excluded:
        print(f"Label-attribution gate: excluding {sorted(excluded)} "
              f"(rates {excluded})", flush=True)
    years = detect_operational_years(dd, cfg)
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
        """Deployment-faithful test protocol: every LIVE-trained model is scored
        only on years strictly AFTER its training data — a deployed system can
        never be tested on its own past. The benchmark model (trained 2010-2012)
        may use every detected year (all are 2013+)."""
        if model_name == "Live-trained":
            return [s for s in op_sets if int(s) > 2014]
        if model_name == MULTI:
            cut = max(multi_train_years) if multi_train_years else 2014
            return [s for s in op_sets if int(s) > cut]
        return op_sets

    # Self-correction experiment (mirrors `python -m solarflare.recalibrate`,
    # in-memory only): recalibrate each model's threshold on the EARLIEST of its
    # valid operational years, then freeze that threshold for the strictly-later
    # years. Uses each model's own valid-year protocol, so it inherits the
    # no-backwards-testing rule.
    recal = {}
    for m in models:
        after = _valid_sets(m)
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
        leaky = (m == MULTI and multi_bench_leaky)
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
        row = {
            "name": m, "detail": details[m],
            "operational_years_used": [int(s) for s in sets_m],
            "operational_tss": round(op_peak, 3),
            "operational_ci": _ci(op_peak_boot, level),
            "by_year": by_year,
        }
        if leaky:
            # No leakage-free benchmark cell exists — operational columns only.
            row.update({
                "benchmark_tss": None, "gap": None,
                "benchmark_ci": None, "gap_ci": None,
                "benchmark_note": (
                    "no leakage-free benchmark test exists for this model: its "
                    "live training years (JSOC "
                    + "+".join(map(str, multi_train_years)) +
                    ") cover the SWAN-SF test split's era, so the same active "
                    "regions appear in training and test"),
                "frozen": {
                    "threshold": round(thr[m], 3),
                    "benchmark_tss": None, "benchmark_ci": None,
                    "operational_tss": round(op_frozen, 3),
                    "operational_ci": _ci(op_frozen_boot, level),
                    "gap": None, "gap_ci": None,
                },
            })
        else:
            yb, pb = y_by_set["benchmark"], p_by_set_model["benchmark"][m]
            b_peak = peak_tss(yb, pb)
            b_frozen = frozen_tss(yb, pb, thr[m])
            gap_boot = boot["benchmark"][m]["peak"] - op_peak_boot
            gap_frozen_boot = boot["benchmark"][m]["frozen"] - op_frozen_boot
            row.update({
                # Backward-compatible headline keys (peak TSS, mean over years):
                "benchmark_tss": round(b_peak, 3),
                "gap": round(b_peak - op_peak, 3),
                "benchmark_ci": _ci(boot["benchmark"][m]["peak"], level),
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
                "_gap_boot": gap_boot,            # internal, stripped before writing
                "_gap_frozen_boot": gap_frozen_boot,
            })
        # Self-corrected deployment: the recalibrated threshold frozen on the
        # strictly-later years, with a PAIRED recovery CI (same replicates).
        # Gap-vs-benchmark fields exist only for models with a leakage-free
        # benchmark cell; the leaky (multi-year) model reports recovery and
        # operational skill only.
        row["recalibrated"] = None
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
            rblock = {
                "calibrated_on": r_info["cal_year"],
                "threshold": round(r_info["threshold"], 3),
                "eval_years": [int(s) for s in ev_yrs],
                "operational_tss": round(recal_op, 3),
                "operational_ci": _ci(recal_op_boot, level),
                "before_tss_same_years": round(before, 3),
                "recovery": round(recal_op - before, 3),
                "recovery_ci": _ci(recal_op_boot - before_boot, level),
            }
            if not leaky:
                yb_, pb_ = y_by_set["benchmark"], p_by_set_model["benchmark"][m]
                bfz = frozen_tss(yb_, pb_, thr[m])
                rblock.update({
                    # Before/after gaps on the SAME eval years (the headline gap
                    # averages more/earlier years — juxtaposing it with gap_after
                    # would conflate the year-set change with the recalibration).
                    "gap_before_same_years": round(bfz - before, 3),
                    "gap_before_same_years_ci": _ci(boot["benchmark"][m]["frozen"] - before_boot, level),
                    "gap_after": round(bfz - recal_op, 3),
                    "gap_after_ci": _ci(boot["benchmark"][m]["frozen"] - recal_op_boot, level),
                })
            else:
                rblock["benchmark_note"] = ("no leakage-free benchmark cell for this "
                                            "model — recovery and operational skill only")
            rblock["by_year"] = by_year_recal
            row["recalibrated"] = rblock
        rows.append(row)

    bench = next(r for r in rows if r["name"] == "Benchmark-trained")
    live = next(r for r in rows if r["name"] == "Live-trained")
    # Does live training close the gap? Compare the two models' gaps ON THE SAME
    # operational years (the live model's valid years, a subset of the benchmark
    # model's — comparing gaps averaged over different year sets would be
    # apples-to-oranges): > 0 means live training genuinely shrinks the gap.
    live_years = [str(y) for y in live["operational_years_used"]]
    bench_op_same_boot = np.mean(
        [boot[s]["Benchmark-trained"]["peak"] for s in live_years], axis=0)
    bench_gap_same_boot = (boot["benchmark"]["Benchmark-trained"]["peak"]
                           - bench_op_same_boot)
    closes_boot = bench_gap_same_boot - live["_gap_boot"]
    closes_ci = _ci(closes_boot, level)
    bench_gap_same = round(bench["benchmark_tss"] - float(
        np.mean([bench["by_year"][s]["peak_tss"] for s in live_years])), 3)

    # Does MORE live data help? The multi-year model has no leakage-free
    # benchmark cell (see the leakage guard), so gap-vs-gap is not defined for
    # it; the meaningful comparison is OPERATIONAL skill on identical years,
    # paired via the shared bootstrap replicates.
    more_data = None
    multi = next((r for r in rows if r["name"] == MULTI), None)
    if multi is not None:
        yrs = [str(y) for y in multi["operational_years_used"]]
        single_op_boot = np.mean([boot[s]["Live-trained"]["peak"] for s in yrs], axis=0)
        multi_op_boot = np.mean([boot[s][MULTI]["peak"] for s in yrs], axis=0)
        diff_boot = multi_op_boot - single_op_boot            # >0 => more data helps
        single_op = float(np.mean(
            [live["by_year"][s]["peak_tss"] if s in live["by_year"]
             else peak_tss(y_by_set[s], p_by_set_model[s]["Live-trained"])
             for s in yrs]))
        more_data = {
            "years_compared": [int(s) for s in yrs],
            "single_year_operational_tss": round(single_op, 3),
            "multi_year_operational_tss": multi["operational_tss"],
            "more_data_improves_operational_skill": bool(
                multi["operational_tss"] > round(single_op, 3)),
            "multi_minus_single_operational_ci": _ci(diff_boot, level),
            "note": ("operational-skill comparison on identical years; no "
                     "leakage-free benchmark cell exists for the multi-year "
                     "model, so gap-vs-gap is not defined for it"),
        }
    # Self-correction findings. "Closed" requires the POINT estimate at/below
    # zero — a CI that merely spans zero is absence of evidence, not evidence of
    # absence, and is reported separately as not-significant. The leaky
    # (multi-year) model has no gap fields; its combined fix reports recovery
    # and operational skill only.
    recal_findings = None
    rb = bench.get("recalibrated")
    if rb:
        recal_findings = {
            "benchmark_trained": {
                "recovery": rb["recovery"], "recovery_ci": rb["recovery_ci"],
                "gap_before_same_years": rb.get("gap_before_same_years"),
                "gap_after": rb.get("gap_after"), "gap_after_ci": rb.get("gap_after_ci"),
                "recovers_significantly": bool(rb["recovery_ci"][0] > 0),
            },
        }
        if multi is not None and multi.get("recalibrated"):
            mb = multi["recalibrated"]
            has_gap = mb.get("gap_after") is not None
            recal_findings["combined_fix"] = {
                "description": "multi-year live training + one-year threshold recalibration",
                "operational_tss": mb["operational_tss"],
                "operational_ci": mb["operational_ci"],
                "recovery": mb["recovery"], "recovery_ci": mb["recovery_ci"],
                "recovers_significantly": bool(mb["recovery_ci"][0] > 0),
                "gap_before_same_years": mb.get("gap_before_same_years"),
                "gap_after": mb.get("gap_after"), "gap_after_ci": mb.get("gap_after_ci"),
                "gap_closed": bool(mb["gap_after"] <= 0) if has_gap else None,
                "gap_not_significant": (bool(mb["gap_after_ci"][0] <= 0 <= mb["gap_after_ci"][1])
                                        if mb.get("gap_after_ci") else None),
                "note": mb.get("benchmark_note"),
            }

    for r in rows:
        r.pop("_gap_boot", None), r.pop("_gap_frozen_boot", None)

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
            "test_protocol": "every live-trained model is scored only on "
                             "operational years strictly after its training data "
                             "(never backwards in time); cross-model gap "
                             "comparisons use identical year sets",
            "leakage_guard": "a model whose live training years overlap the "
                             "benchmark test split's era gets no benchmark/gap "
                             "cells (same active regions would appear in "
                             "training and test) — operational columns only",
            "climatology_note": "a constant-probability (climatology) forecast has "
                                "TSS = 0 by construction — the zero line IS the "
                                "climatology baseline",
            "recalibration": "self-correction experiment: each model's threshold is "
                             "re-fit on the earliest of its valid operational years "
                             "and frozen for the strictly-later years; recovery CIs "
                             "are paired (frozen2 vs frozen on the same cluster-"
                             "bootstrap replicates)",
            "label_gate": "years whose measured HEK AR-attribution rate falls "
                          "below scorecard.min_label_attribution are excluded "
                          "from all label-dependent scoring — their labels "
                          "under-count positives (see test_sets.excluded_years)",
        },
        "test_sets": {
            "benchmark": f"held-out SWAN-SF magnetograms ({int(len(y_by_set['benchmark']))} windows, "
                         f"{int(y_by_set['benchmark'].sum())} flares)",
            "operational": "live JSOC, unseen year(s): " + ", ".join(
                f"{s} ({int(len(y_by_set[s]))} windows, {int(y_by_set[s].sum())} flares)"
                for s in op_sets),
            "operational_years": [int(s) for s in op_sets],
            "excluded_years": {
                str(y): (f"HEK AR-attribution rate {r:.2f} is below the "
                         f"{float((cfg.get('scorecard', {}) or {}).get('min_label_attribution', 0)):.2f} gate — "
                         "labels under-count positives (a flare without a NOAA AR "
                         "number can never mark a positive window), so TSS on this "
                         "year would measure catalog decay, not model skill")
                for y, r in sorted(excluded.items())},
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
            "live_training_closes_gap": bool(live["gap"] < bench_gap_same),
            "gap_comparison_years": [int(s) for s in live_years],
            "benchmark_gap_on_comparison_years": bench_gap_same,
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
