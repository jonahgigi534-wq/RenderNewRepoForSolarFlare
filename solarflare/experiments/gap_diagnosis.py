"""WHY does the benchmark-vs-operational gap exist? Three testable mechanisms.

The scorecard (solarflare.scorecard) shows benchmark scores overstate operational
skill. This experiment separates the candidate explanations instead of guessing:

  1. LABEL-PROTOCOL MISMATCH — SWAN-SF's FL/NF labels and our HEK/SWPC-derived
     labels are built by different pipelines. If they disagree on the SAME
     windows, part of the "gap" is measuring label noise, not model failure.
     Test: re-label every SWAN-SF partition-1 window with OUR labeler (M+ flare
     from that NOAA AR within 24 h of window end, HEK/SWPC catalog) and report
     the agreement/confusion.

  2. DISTRIBUTION SHIFT — the operational year may simply look different (solar-
     cycle phase, instrument drift, region mix). Test: two-sample KS statistic
     per feature between the benchmark test windows and the operational year.

  3. MODEL DIVERGENCE — if benchmark-trained and live-trained models rely on
     DIFFERENT features, they learned different (dataset-specific) shortcuts.
     Test: permutation importance for both models on the same operational data,
     plus the rank correlation between their importance orderings, annotated
     with the known physics of the top SHARP parameters.

Trains in-memory only (save=False); the deployed model is untouched.
Writes gap_diagnosis.json (project root) for /api/diagnosis + the dashboard.

Run:  python -m solarflare.experiments.gap_diagnosis
"""
from __future__ import annotations

import json
import os
from datetime import timedelta

import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.inspection import permutation_importance

from .. import data as dataio
from .. import sharpdata, sharptrain
from ..config import load_config

# What the top SHARP parameters physically mean (flare-prediction literature,
# e.g. Bobra & Couvidat 2015; Schrijver 2007). Shown next to the rankings so the
# importance table reads as physics, not feature indices.
PHYSICS = {
    "R_VALUE": "unsigned flux near polarity-inversion lines (Schrijver's R) — classic flare precursor",
    "TOTUSJH": "total unsigned current helicity — a top predictor in Bobra & Couvidat (2015)",
    "TOTPOT": "proxy for total photospheric magnetic free energy",
    "TOTUSJZ": "total unsigned vertical current",
    "USFLUX": "total unsigned magnetic flux (region size × field strength)",
    "ABSNJZH": "absolute net current helicity",
    "SAVNCPP": "sum of |net current| per polarity",
    "MEANSHR": "mean magnetic shear angle",
    "SHRGT45": "fraction of field sheared beyond 45°",
    "MEANGAM": "mean inclination angle",
}


def _keyword_of(feature_name: str) -> str:
    return feature_name.split("__")[0]


# ----------------------------------------------------------------------
# 1. Label-protocol audit
# ----------------------------------------------------------------------
HARP_NOAA_URL = "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"


def harp_to_noaa(cache_dir: str, *, verbose: bool = True) -> dict:
    """JSOC's official HARPNUM -> [NOAA AR numbers] mapping (SWAN-SF instance
    filenames carry the HARP number, while flare catalogs use NOAA numbers)."""
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, "harp_to_noaa.txt")
    if not os.path.exists(local):
        if verbose:
            print(f"  fetching {HARP_NOAA_URL} ...", flush=True)
        r = requests.get(HARP_NOAA_URL, timeout=60)
        r.raise_for_status()
        with open(local, "w", encoding="utf-8") as fh:
            fh.write(r.text)
    out: dict[int, list[int]] = {}
    with open(local, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            try:
                out[int(parts[0])] = [int(a) for a in parts[1].split(",") if a.strip().isdigit()]
            except ValueError:
                continue
    return out


def label_audit(dp: dict, cfg: dict, *, obs_h: int = 12, pred_h: int = 24,
                verbose: bool = True) -> dict:
    """Re-label SWAN-SF windows with OUR HEK/SWPC labeler and compare.
    SWAN-SF's `ar<N>` is the HARP number, so windows are mapped to their NOAA
    AR number(s) via JSOC's official table first."""
    groups, y = dp["groups"], dp["y"]
    # swansf_data stores the window START time (from the instance filename);
    # the prediction window opens at start + obs_h.
    starts = dp["end_times"]
    t_min, t_max = min(starts), max(starts)
    if verbose:
        print(f"Label audit: SWAN-SF windows span {t_min.date()} .. {t_max.date()}")
        print("Fetching the HEK/SWPC M+ flare catalog for that span ...", flush=True)
    by_ar = sharpdata.fetch_flares(t_min - timedelta(days=1),
                                   t_max + timedelta(hours=obs_h + pred_h + 2),
                                   verbose=verbose)
    mapping = harp_to_noaa(os.path.join(cfg["_project_root"], "data", "sharp_live"),
                           verbose=verbose)
    both = swan_only = hek_only = neither = skipped = 0
    for harp, t0, lab in zip(groups, starts, y):
        noaa_ars = mapping.get(int(harp), [])
        if not noaa_ars:
            skipped += 1
            continue
        end = t0 + timedelta(hours=obs_h)
        ours = 1 if any(sharpdata._flare_within(by_ar, ar, end, end + timedelta(hours=pred_h))
                        for ar in noaa_ars) else 0
        if lab and ours:
            both += 1
        elif lab and not ours:
            swan_only += 1
        elif ours and not lab:
            hek_only += 1
        else:
            neither += 1
    n = both + swan_only + hek_only + neither
    agree = (both + neither) / n if n else 0.0
    # Positive-class agreement is the number that matters (99% of windows are
    # negative in both, which inflates plain agreement).
    pos_union = both + swan_only + hek_only
    pos_jaccard = both / pos_union if pos_union else 0.0
    return {
        "n_windows_compared": n,
        "n_skipped_no_noaa_mapping": int(skipped),
        "agreement": round(agree, 4),
        "positive_jaccard": round(pos_jaccard, 4),
        "confusion": {"both_positive": both, "swansf_only_positive": swan_only,
                      "hek_only_positive": hek_only, "both_negative": neither},
        "reading": ("positive_jaccard is the share of flare-windows BOTH protocols "
                    "agree on; swansf_only/hek_only counts are windows where the two "
                    "labeling pipelines contradict each other — that disagreement is "
                    "label noise the operational test inherits."),
    }


# ----------------------------------------------------------------------
# 2. Distribution shift (feature-space KS)
# ----------------------------------------------------------------------
def distribution_shift(Xb: np.ndarray, Xo: np.ndarray, names: list[str],
                       top_k: int = 15) -> dict:
    """Two-sample KS per feature (NaN-stripped), plus a per-keyword rollup."""
    ks = []
    for j, nm in enumerate(names):
        a = Xb[:, j][np.isfinite(Xb[:, j])]
        b = Xo[:, j][np.isfinite(Xo[:, j])]
        if len(a) < 50 or len(b) < 50:
            continue
        stat = float(ks_2samp(a, b).statistic)
        ks.append((nm, stat))
    ks.sort(key=lambda r: r[1], reverse=True)
    by_kw: dict[str, float] = {}
    for nm, stat in ks:
        kw = _keyword_of(nm)
        by_kw[kw] = max(by_kw.get(kw, 0.0), stat)
    kw_sorted = sorted(by_kw.items(), key=lambda r: r[1], reverse=True)
    return {
        "metric": "two-sample Kolmogorov-Smirnov statistic (0 = identical "
                  "distributions, 1 = disjoint)",
        "median_ks_all_features": round(float(np.median([s for _, s in ks])), 3),
        "top_shifted_features": [{"feature": nm, "ks": round(s, 3)} for nm, s in ks[:top_k]],
        "max_ks_by_keyword": [{"keyword": k, "ks": round(s, 3),
                               "physics": PHYSICS.get(k, "")} for k, s in kw_sorted],
    }


# ----------------------------------------------------------------------
# 3. Model divergence (permutation importance on the SAME operational data)
# ----------------------------------------------------------------------
def importance_comparison(models: dict, Xo: np.ndarray, yo: np.ndarray,
                          names: list[str], rng, *, n_sample: int = 4000,
                          verbose: bool = True) -> dict:
    idx = rng.choice(len(yo), size=min(n_sample, len(yo)), replace=False)
    Xs, ys = Xo[idx], yo[idx]
    per_model = {}
    vectors = {}
    for mname, payload in models.items():
        if verbose:
            print(f"Permutation importance: {mname} ...", flush=True)
        r = permutation_importance(payload["model"], Xs, ys, scoring="roc_auc",
                                   n_repeats=5, random_state=42, n_jobs=-1)
        imp = r.importances_mean
        vectors[mname] = imp
        by_kw: dict[str, float] = {}
        for j, nm in enumerate(names):
            kw = _keyword_of(nm)
            by_kw[kw] = by_kw.get(kw, 0.0) + float(imp[j])
        top = sorted(by_kw.items(), key=lambda r: r[1], reverse=True)
        per_model[mname] = [{"keyword": k, "importance": round(v, 5),
                             "physics": PHYSICS.get(k, "")} for k, v in top[:8]]
    m1, m2 = list(vectors)
    rho = float(spearmanr(vectors[m1], vectors[m2]).statistic)
    return {
        "note": "importance measured on the SAME operational windows for both "
                "models (roc_auc drop when a feature is permuted; keyword rollup "
                "sums its 7 summary stats)",
        "n_windows_used": int(len(ys)),
        "top_keywords": per_model,
        "spearman_rank_correlation": round(rho, 3),
        "reading": ("rank correlation near 1 = both models learned the same "
                    "physics; low correlation = at least one model leans on "
                    "dataset-specific shortcuts that do not transfer."),
    }


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    root = cfg["_project_root"]
    dd = os.path.join(root, "data", "sharp_live")
    rng = np.random.default_rng(42)
    names = sharptrain.feature_names_live(cfg)

    print("Training both models in-memory (deployed model untouched) ...", flush=True)
    v_bench = sharptrain.train(os.path.join(dd, "dataset_swansf_p1.npz"), cfg, save=False)
    v_live = sharptrain.train(os.path.join(dd, "dataset_2014.npz"), cfg, save=False)
    models = {"Benchmark-trained": v_bench, "Live-trained": v_live}

    dp = sharpdata.load_dataset(os.path.join(dd, "dataset_swansf_p1.npz"))
    do = sharpdata.load_dataset(os.path.join(dd, "dataset_2015.npz"))
    _, _, ite = sharptrain.time_group_split(dp["groups"], dp["end_times"],
                                            cfg["sharp_live"]["split"])
    Xb = dataio.build_matrix(dp["X3d"][ite], cfg)
    Xo, yo = dataio.build_matrix(do["X3d"], cfg), do["y"]

    print("\n[1/3] Label-protocol audit (SWAN-SF labels vs our HEK labeler) ...", flush=True)
    audit = label_audit(dp, cfg)
    print(f"  agreement={audit['agreement']:.3f}  positive-jaccard={audit['positive_jaccard']:.3f}")

    print("\n[2/3] Distribution shift (benchmark test vs operational 2015) ...", flush=True)
    shift = distribution_shift(Xb, Xo, names)
    print(f"  median KS={shift['median_ks_all_features']}  "
          f"top: {shift['top_shifted_features'][0]}")

    print("\n[3/3] Model divergence (permutation importance) ...", flush=True)
    diverge = importance_comparison(models, Xo, yo, names, rng)
    print(f"  Spearman rank correlation = {diverge['spearman_rank_correlation']}")

    base_rates = {
        "benchmark_test": round(float(dp["y"][ite].mean()), 4),
        "operational_2015": round(float(yo.mean()), 4),
    }
    out = {
        "title": "Why does the benchmark-operational gap exist?",
        "question": "label-protocol mismatch, distribution shift, or model divergence?",
        "base_rates": base_rates,
        "label_audit": audit,
        "distribution_shift": shift,
        "model_divergence": diverge,
    }
    path = os.path.join(root, "gap_diagnosis.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {path}", flush=True)
    return out


def main():
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
