"""Train the LIVE SHARP flare model on the JSOC+HEK dataset.

The whole point: a self-contained pipeline we own end to end. The saved model is
a sklearn Pipeline  median-impute -> StandardScaler -> classifier  (optionally
isotonic-calibrated), so live JSOC data goes through the IDENTICAL transform the
model trained on — no mystery external normalization. Leakage-free, region-disjoint
chronological split (a whole active region is in exactly one of train/val/test).

    python -m solarflare.sharptrain --data data/sharp_live/dataset_2014.npz

Artifacts: models/flare_sharp_live_model.joblib (+ .meta.json).
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as dataio
from . import evaluate as ev
from . import sharpdata
from .config import load_config


def feature_names_live(cfg: dict) -> list[str]:
    names = []
    for ch in cfg["sharp_live"]["keywords"]:
        for s in cfg["sharp_live"]["summary_stats"]:
            names.append(f"{ch}__{s}")
    return names


def build_candidates(cfg: dict) -> dict:
    """Each candidate is a full pipeline: impute -> scale -> classifier, so the
    saved model carries its own preprocessing (the 'recipe')."""
    rs = int(cfg["sharp_live"].get("random_state", 42))
    req = set(cfg["sharp_live"].get("candidates", []))

    def pre():
        return [SimpleImputer(strategy="median"), StandardScaler()]

    zoo = {}
    if "logistic" in req:
        zoo["logistic"] = make_pipeline(
            *pre(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if "random_forest" in req:
        zoo["random_forest"] = make_pipeline(
            *pre(), RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                           class_weight="balanced", random_state=rs))
    if "extra_trees" in req:
        zoo["extra_trees"] = make_pipeline(
            *pre(), ExtraTreesClassifier(n_estimators=300, n_jobs=-1,
                                         class_weight="balanced", random_state=rs))
    if "hist_gradient_boosting" in req:
        zoo["hist_gradient_boosting"] = make_pipeline(
            *pre(), HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=400, l2_regularization=1.0,
                early_stopping=True, class_weight="balanced", random_state=rs))
    return zoo


def time_group_split(groups, end_times, frac: dict):
    """Region-disjoint chronological split: order regions by their median time and
    fill train -> val -> test by cumulative sample count. No region spans a boundary
    (kills the main leakage path), and splits are ordered in time."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    med = {}
    for g in uniq:
        idx = np.where(groups == g)[0]
        ts = sorted(end_times[i] for i in idx)
        med[g] = ts[len(ts) // 2]
    regions_sorted = sorted(uniq.tolist(), key=lambda g: med[g])
    n = len(groups)
    tr_cut = frac["train"] * n
    val_cut = (frac["train"] + frac["val"]) * n
    idx_tr, idx_val, idx_te, cum = [], [], [], 0
    for g in regions_sorted:
        idx = np.where(groups == g)[0]
        bucket = idx_tr if cum < tr_cut else idx_val if cum < val_cut else idx_te
        bucket.extend(idx.tolist())
        cum += len(idx)
    return (np.array(idx_tr, dtype=int), np.array(idx_val, dtype=int),
            np.array(idx_te, dtype=int))


def train(data_path: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    sl = cfg["sharp_live"]
    sel = sl["selection_metric"]
    print(f"Loading dataset: {data_path}")
    d = sharpdata.load_dataset(data_path)
    X3d, y, groups, ends = d["X3d"], d["y"], d["groups"], d["end_times"]
    Xf = dataio.build_matrix(X3d, cfg)                    # (n, 17*7 = 119) raw-derived features
    print(f"  samples={len(y)}  features={Xf.shape[1]}  positives={int(y.sum())} "
          f"({y.mean():.3%})")

    itr, iva, ite = time_group_split(groups, ends, sl["split"])
    Xtr, ytr = Xf[itr], y[itr]
    Xva, yva = Xf[iva], y[iva]
    Xte, yte = Xf[ite], y[ite]
    print(f"  split  train={len(ytr)} ({ytr.mean():.2%}+)  val={len(yva)} "
          f"({yva.mean():.2%}+)  test={len(yte)} ({yte.mean():.2%}+)")
    if ytr.sum() == 0 or yva.sum() == 0 or yte.sum() == 0:
        raise RuntimeError("a split has zero positives — widen the date span")

    score_fn = {"tss": ev.tss, "hss": ev.hss}[sel]
    print("\n=== Bake-off (ranked on validation) ===")
    results = []
    for name, est in build_candidates(cfg).items():
        t0 = time.time()
        try:
            est.fit(Xtr, ytr)
            pv = est.predict_proba(Xva)[:, 1]
            thr, _ = ev.best_threshold(yva, pv, sel)
            sc = score_fn(yva, (pv >= thr).astype(int))
            results.append((name, sc))
            print(f"  {name:24s} val {sel.upper()}={sc:.3f}  ({time.time()-t0:.0f}s)")
        except Exception as exc:                          # noqa: BLE001
            print(f"  {name:24s} FAILED: {type(exc).__name__}: {exc}")
    if not results:
        raise RuntimeError("no candidate trained")
    results.sort(key=lambda r: r[1], reverse=True)
    winner = results[0][0]
    print(f"  -> winner: {winner} (val {sel.upper()}={results[0][1]:.3f})")

    print(f"\nCalibrating {winner} (isotonic) ...")
    model = CalibratedClassifierCV(build_candidates(cfg)[winner], method="isotonic", cv=3)
    model.fit(Xtr, ytr)
    pval = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    ops = ev.operating_points(yva, pval, min_recall=sl["high_precision_min_recall"])
    for nm, op in ops.items():
        op["test"] = ev.full_report(yte, (pte >= op["threshold"]).astype(int))

    default_op = "balanced"
    print("\n=== Operating points on held-out TEST ===")
    print(f"  {'point':16s} {'thr':>5s} {'TSS':>6s} {'recall':>7s} {'prec':>6s} {'F1':>6s}")
    for nm in ("high_recall", "balanced", "high_precision"):
        r = ops[nm]["test"]
        star = " *" if nm == default_op else ""
        print(f"  {nm:16s} {ops[nm]['threshold']:5.3f} {r['tss']:6.3f} "
              f"{r['recall']:7.3f} {r['precision']:6.3f} {r['f1']:6.3f}{star}")

    payload = {
        "model": model,
        "kind": "live_sharp_jsoc",
        "winner": winner,
        "bakeoff": [{"model": n, f"val_{sel}": round(s, 4)} for n, s in results],
        "threshold": ops[default_op]["threshold"],
        "default_operating_point": default_op,
        "operating_points": {k: {"threshold": v["threshold"], "test": v["test"]}
                             for k, v in ops.items()},
        "keywords": sl["keywords"],
        "summary_stats": sl["summary_stats"],
        "feature_names": feature_names_live(cfg),
        "observation_window_h": sl["observation_window_h"],
        "prediction_window_h": sl["prediction_window_h"],
        "n_samples": int(len(y)),
        "base_rate": float(y.mean()),
        "data_span": [str(min(ends)), str(max(ends))],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": ops[default_op]["test"],
        "sklearn_version": __import__("sklearn").__version__,
        "version": __import__("solarflare").__version__,
    }
    out = sl.get("model_path", "models/flare_sharp_live_model.joblib")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    joblib.dump(payload, out)
    meta = {k: v for k, v in payload.items() if k != "model"}
    with open(out.replace(".joblib", ".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"\nSaved live SHARP model -> {out} (+ .meta.json)")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Train the live SHARP flare model.")
    ap.add_argument("--data", default="data/sharp_live/dataset_2014.npz")
    args = ap.parse_args()
    train(args.data)


if __name__ == "__main__":
    main()
