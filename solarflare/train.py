"""Train the SHARP solar-flare model and persist it.

What it does now (a model *bake-off*, not a single fit):
  1. Trains every available candidate (HistGB, RandomForest, ExtraTrees,
     Logistic, LightGBM, XGBoost) on the augmented train partitions.
  2. Ranks them on the VALIDATION partition by `selection_metric` (TSS).
  3. Calibrates the winner (isotonic) so probabilities are trustworthy.
  4. Fits THREE operating points on validation — high_recall / balanced /
     high_precision — and reports each on the CLEAN TEST partition.
  5. Saves the winner + all operating points + provenance.

Usage
-----
    python -m solarflare.train               # real SWAN-SF .pkl data
    python -m solarflare.train --synthetic   # built-in fixture (no download)

Artifacts (models/): flare_sharp_model.joblib (+ .pkl, .meta.json)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from . import data as dataio
from . import evaluate as ev
from .config import load_config


# ----------------------------------------------------------------------
# Candidate model zoo (lightgbm / xgboost added only if importable)
# ----------------------------------------------------------------------
def build_candidates(cfg: dict) -> dict:
    rs = cfg["training"]["random_state"]
    requested = set(cfg["training"].get("candidates", []))
    zoo = {}
    if "hist_gradient_boosting" in requested:
        zoo["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=rs)
    if "random_forest" in requested:
        zoo["random_forest"] = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=rs)
    if "extra_trees" in requested:
        zoo["extra_trees"] = ExtraTreesClassifier(
            n_estimators=200, n_jobs=-1, random_state=rs)
    if "logistic" in requested:
        zoo["logistic"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"))
    if "lightgbm" in requested:
        try:
            from lightgbm import LGBMClassifier
            zoo["lightgbm"] = LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                random_state=rs, verbosity=-1)
        except ImportError:
            pass
    if "xgboost" in requested:
        try:
            from xgboost import XGBClassifier
            zoo["xgboost"] = XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                eval_metric="logloss", n_jobs=-1, random_state=rs)
        except ImportError:
            pass
    return zoo


def _fit(est, X, y, sw):
    """Fit with balanced sample weights; pipelines fall back to class_weight."""
    try:
        est.fit(X, y, sample_weight=sw)
    except (TypeError, ValueError):
        est.fit(X, y)
    return est


def _proba(est, X):
    return est.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------
def train(synthetic: bool = False, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    tcfg = cfg["training"]
    sel = tcfg["selection_metric"]
    print("Loading data ...", "(synthetic fixture)" if synthetic else "(SWAN-SF .pkl)")
    if synthetic:
        (Xtr, ytr), (Xval, yval), (Xte, yte) = dataio.load_synthetic_split(cfg)
    else:
        (Xtr, ytr), (Xval, yval), (Xte, yte) = dataio.load_split(cfg)
    print(f"  train={Xtr.shape}  val={Xval.shape}  test={Xte.shape}  "
          f"(train pos-rate {ytr.mean():.3f})")

    sw = compute_sample_weight("balanced", ytr)
    score_fn = {"tss": ev.tss, "hss": ev.hss}[sel]

    # ---- bake-off -----------------------------------------------------
    print("\n=== Model bake-off (ranked on validation) ===")
    results = []
    for name, est in build_candidates(cfg).items():
        t0 = time.time()
        try:
            _fit(est, Xtr, ytr, sw)
            pv = _proba(est, Xval)
            thr, _ = ev.best_threshold(yval, pv, sel)
            val_skill = score_fn(yval, (pv >= thr).astype(int))
            results.append((name, val_skill, time.time() - t0))
            print(f"  {name:24s}  val {sel.upper()}={val_skill:.3f}   "
                  f"({time.time()-t0:.0f}s)")
        except Exception as exc:                       # noqa: BLE001
            print(f"  {name:24s}  FAILED: {type(exc).__name__}: {exc}")
    if not results:
        raise RuntimeError("no candidate model trained successfully")
    results.sort(key=lambda r: r[1], reverse=True)
    winner_name = results[0][0]
    print(f"  -> winner: {winner_name}  (val {sel.upper()}={results[0][1]:.3f})")

    # ---- calibrate the winner ----------------------------------------
    print(f"\nCalibrating {winner_name} (isotonic) ...")
    fresh = build_candidates(cfg)[winner_name]
    model = CalibratedClassifierCV(fresh, method="isotonic", cv=3)
    try:
        model.fit(Xtr, ytr, sample_weight=sw)
    except (TypeError, ValueError):
        model.fit(Xtr, ytr)

    # ---- operating points (fit on val, evaluated on test) ------------
    pval = model.predict_proba(Xval)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    ops = ev.operating_points(yval, pval, min_recall=tcfg["high_precision_min_recall"])
    for name, op in ops.items():
        op["test"] = ev.full_report(yte, (pte >= op["threshold"]).astype(int))

    default_op = cfg["live"]["operating_point"]
    if default_op not in ops:
        default_op = "balanced"

    print("\n=== Operating points on CLEAN TEST partition ===")
    print(f"  {'point':16s} {'thr':>5s} {'TSS':>6s} {'recall':>7s} "
          f"{'prec':>6s} {'F1':>6s}")
    for name in ("high_recall", "balanced", "high_precision"):
        r = ops[name]["test"]
        star = " *" if name == default_op else ""
        print(f"  {name:16s} {ops[name]['threshold']:5.3f} {r['tss']:6.3f} "
              f"{r['recall']:7.3f} {r['precision']:6.3f} {r['f1']:6.3f}{star}")
    print(f"  (* = default used by the live predictor; change via "
          f"live.operating_point)")

    # ---- persist ------------------------------------------------------
    # Headline metrics = the operating point actually used live (so /health and
    # the UI report what the running forecast does, not a different threshold).
    headline = ops[default_op]["test"]
    payload = {
        "model": model,
        "winner": winner_name,
        "bakeoff": [{"model": n, f"val_{sel}": round(s, 4)} for n, s, _ in results],
        "threshold": ops[default_op]["threshold"],
        "default_operating_point": default_op,
        "operating_points": {k: {"threshold": v["threshold"], "test": v["test"]}
                             for k, v in ops.items()},
        "forecast_bands": cfg.get("forecast_bands", []),
        "feature_names": dataio.feature_names(cfg),
        "task": tcfg["task"],
        "classes": cfg["classes"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": synthetic,
        "metrics": headline,                   # headline = the live operating point (default: balanced)
        "sklearn_version": __import__("sklearn").__version__,
        "version": __import__("solarflare").__version__,
    }

    out_dir = cfg["paths"]["model_dir"]
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(payload, os.path.join(out_dir, "flare_sharp_model.joblib"))
    import pickle
    with open(os.path.join(out_dir, "flare_sharp_model.pkl"), "wb") as fh:
        pickle.dump(payload, fh)
    meta = {k: v for k, v in payload.items() if k != "model"}
    with open(os.path.join(out_dir, "flare_sharp_model.meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"\nSaved {winner_name} -> {out_dir}\\flare_sharp_model.joblib (+ .pkl, .meta.json)")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Train the SHARP flare model (bake-off).")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on the built-in synthetic fixture (no download)")
    args = ap.parse_args()
    train(synthetic=args.synthetic)


if __name__ == "__main__":
    main()
