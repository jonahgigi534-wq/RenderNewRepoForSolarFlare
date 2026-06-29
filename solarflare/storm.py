"""Geomagnetic-storm forecaster — the scientific core of Part 2.

Real machine learning (mirrors the flare trainer in train.py), kept in its own
module and feature space: it forecasts P(Kp >= threshold storm in the next N
hours) from the L1 solar wind. Honest, leakage-free protocol:

  * Train on NASA OMNI hourly (stormdata.load_split): solar-wind history -> Kp.
  * CHRONOLOGICAL split with a multi-day gap (storms autocorrelate for many
    hours; random shuffling would leak the answer). Never shuffled.
  * Bake-off ranked on a held-out VALIDATION slice by TSS; winner isotonic-
    calibrated; three operating points frozen and reported on a CLEAN TEST slice.
  * Headline metric is TSS/HSS (reuses evaluate.py), not accuracy.

Live inference reconstructs the model's channels from the real-time L1 feeds and
adds NOAA's own Kp forecast as a corroboration track. Resilient: degrades to a
climatological base rate if a source or the model is missing. Never raises.

Usage
-----
    python -m solarflare.storm               # train on real OMNI (HAPI)
    python -m solarflare.storm --synthetic   # offline fixture (no download)
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
from sklearn.utils.class_weight import compute_sample_weight

from . import evaluate as ev
from . import geomag, stormdata, train
from .config import load_config

_MODEL_CACHE = None
_MODEL_FILE = "storm_kp_model"


def _build_candidates(cfg: dict) -> dict:
    """Reuse the flare model zoo, parameterised by the storm config."""
    return train.build_candidates({"training": cfg["storm"]})


# ----------------------------------------------------------------------
# Training (mirrors train.train)
# ----------------------------------------------------------------------
def train_model(synthetic: bool = False, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    sc = cfg["storm"]
    sel = sc["selection_metric"]
    print("Loading OMNI data ...", "(synthetic fixture)" if synthetic else "(NASA OMNI via HAPI)")
    loader = stormdata.load_synthetic_split if synthetic else stormdata.load_split
    (Xtr, ytr), (Xval, yval), (Xte, yte), info = loader(cfg)
    print(f"  train={Xtr.shape}  val={Xval.shape}  test={Xte.shape}  "
          f"(overall storm rate {info['pos_rate']:.3f})")
    if ytr.sum() == 0 or yval.sum() == 0 or yte.sum() == 0:
        raise RuntimeError("a split has no positive storms; widen train_years / window")

    # Per-feature training means (NaN-safe) — persisted so live inference imputes
    # any incomplete-window NaN with the TRAINING distribution, not a batch-size-
    # dependent self-mean. Keeps train/serve feature stats consistent.
    feat_means = np.nanmean(Xtr, axis=0)
    feat_means = np.where(np.isfinite(feat_means), feat_means, 0.0)

    sw = compute_sample_weight("balanced", ytr)
    score_fn = {"tss": ev.tss, "hss": ev.hss}[sel]

    print("\n=== Storm model bake-off (ranked on validation) ===")
    results = []
    for name, est in _build_candidates(cfg).items():
        t0 = time.time()
        try:
            try:
                est.fit(Xtr, ytr, sample_weight=sw)
            except (TypeError, ValueError):
                est.fit(Xtr, ytr)
            pv = est.predict_proba(Xval)[:, 1]
            thr, _ = ev.best_threshold(yval, pv, sel)
            skill = score_fn(yval, (pv >= thr).astype(int))
            results.append((name, skill))
            print(f"  {name:24s}  val {sel.upper()}={skill:.3f}   ({time.time()-t0:.0f}s)")
        except Exception as exc:                               # noqa: BLE001
            print(f"  {name:24s}  FAILED: {type(exc).__name__}: {exc}")
    if not results:
        raise RuntimeError("no storm candidate trained successfully")
    results.sort(key=lambda r: r[1], reverse=True)
    winner = results[0][0]
    print(f"  -> winner: {winner}  (val {sel.upper()}={results[0][1]:.3f})")

    # NOTE: cv=3 is a stratified k-fold without a temporal gap, so the calibration
    # map can be mildly optimistic on autocorrelated data. The headline TEST TSS/HSS
    # below is unaffected (scored by the final model on the gapped test slice).
    print(f"\nCalibrating {winner} (isotonic) ...")
    model = CalibratedClassifierCV(_build_candidates(cfg)[winner], method="isotonic", cv=3)
    try:
        model.fit(Xtr, ytr, sample_weight=sw)
    except (TypeError, ValueError):
        model.fit(Xtr, ytr)

    pval = model.predict_proba(Xval)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    ops = ev.operating_points(yval, pval, min_recall=sc["high_precision_min_recall"])
    for op in ops.values():
        op["test"] = ev.full_report(yte, (pte >= op["threshold"]).astype(int))
    default_op = sc.get("operating_point", "balanced")
    if default_op not in ops:
        default_op = "balanced"

    print("\n=== Operating points on CLEAN TEST slice ===")
    print(f"  {'point':16s} {'thr':>5s} {'TSS':>6s} {'recall':>7s} {'prec':>6s}")
    for name in ("high_recall", "balanced", "high_precision"):
        r = ops[name]["test"]; star = " *" if name == default_op else ""
        print(f"  {name:16s} {ops[name]['threshold']:5.3f} {r['tss']:6.3f} "
              f"{r['recall']:7.3f} {r['precision']:6.3f}{star}")

    payload = {
        "model": model,
        "task": f"kp_ge_{sc['kp_storm_threshold']:g}_in_{sc['prediction_window_h']}h",
        "winner": winner,
        "bakeoff": [{"model": n, f"val_{sel}": round(s, 4)} for n, s in results],
        "threshold": ops[default_op]["threshold"],
        "default_operating_point": default_op,
        "operating_points": {k: {"threshold": v["threshold"], "test": v["test"]}
                             for k, v in ops.items()},
        "feature_names": stormdata.feature_names(cfg),
        "channels": stormdata.CHANNELS,
        "kp_storm_threshold": sc["kp_storm_threshold"],
        "prediction_window_h": sc["prediction_window_h"],
        "observation_window_h": sc["observation_window_h"],
        "base_rate": round(info["pos_rate"], 4),
        "feature_means": feat_means.tolist(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": synthetic,
        "metrics": ops[default_op]["test"],
        "sklearn_version": __import__("sklearn").__version__,
        "version": __import__("solarflare").__version__,
    }
    out_dir = cfg["paths"]["model_dir"]
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(payload, os.path.join(out_dir, f"{_MODEL_FILE}.joblib"))
    import pickle
    with open(os.path.join(out_dir, f"{_MODEL_FILE}.pkl"), "wb") as fh:
        pickle.dump(payload, fh)
    meta = {k: v for k, v in payload.items() if k != "model"}
    with open(os.path.join(out_dir, f"{_MODEL_FILE}.meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"\nSaved {winner} -> {out_dir}\\{_MODEL_FILE}.joblib (+ .pkl, .meta.json)")
    global _MODEL_CACHE
    _MODEL_CACHE = None
    return payload


# ----------------------------------------------------------------------
# Loading + live forecast
# ----------------------------------------------------------------------
def load_storm_model(cfg: dict | None = None):
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    cfg = cfg or load_config()
    path = os.path.join(cfg["paths"]["model_dir"], f"{_MODEL_FILE}.joblib")
    if not os.path.exists(path):
        return None
    try:
        _MODEL_CACHE = joblib.load(path)
    except Exception:                                          # noqa: BLE001
        _MODEL_CACHE = None
    return _MODEL_CACHE


def _band(p: float) -> str:
    return "Severe" if p >= 0.65 else "High" if p >= 0.40 else "Moderate" if p >= 0.15 else "Low"


def _noaa_kp_forecast(cfg: dict) -> dict:
    """NOAA's official predicted Kp (next ~3 days) as a corroboration track."""
    res = sources.get_kp_forecast(cfg)
    rows = res.data
    if not res.ok or not isinstance(rows, list) or len(rows) < 2:
        return {"available": False}
    pred = []
    if isinstance(rows[0], dict):
        for r in rows:
            if str(r.get("observed", "")).lower() == "predicted":
                try: pred.append(float(r.get("kp")))
                except (TypeError, ValueError): pass
    else:
        header = [str(h).lower() for h in rows[0]]
        ki = header.index("kp") if "kp" in header else 1
        oi = header.index("observed") if "observed" in header else 2
        for r in rows[1:]:
            try:
                if str(r[oi]).lower() == "predicted":
                    pred.append(float(r[ki]))
            except (TypeError, ValueError, IndexError):
                pass
    if not pred:
        return {"available": False}
    mx = max(pred)
    return {"available": True, "max_kp_next_3d": round(mx, 2),
            "g_scale": geomag.kp_to_g(mx), "status": res.status}


def storm_forecast(cfg: dict | None = None) -> dict:
    """Assemble the live geomagnetic-storm forecast. Always returns a clean,
    honest result (model track if available; otherwise NOAA + climatology)."""
    cfg = cfg or load_config()
    sc = cfg["storm"]
    model = load_storm_model(cfg)
    try:
        X, summary, status, notes = stormdata.live_feature_vector(cfg)
    except Exception as exc:                                   # never raise to caller
        X, summary, status, notes = None, {}, "unavailable", [f"live feed error: {type(exc).__name__}"]
    try:
        noaa = _noaa_kp_forecast(cfg)
    except Exception:                                          # noqa: BLE001
        noaa = {"available": False}

    ml = {"available": False}
    if model is not None and X is not None:
        try:
            # Impute residual NaN with persisted training means so a partial-data
            # window never reaches predict_proba as NaN (logistic raises; trees
            # would score garbage). If still not finite, skip the ML track cleanly.
            fm = model.get("feature_means")
            if fm is not None:
                X = np.where(np.isnan(X), np.asarray(fm, float), X)
            if not np.isfinite(X).all():
                raise ValueError("incomplete live feature vector")
            p = float(model["model"].predict_proba(X)[:, 1][0])
            thr = model["threshold"]
            ml = {
                "available": True,
                "p_storm": round(p, 4),
                "prediction": "storm likely (Kp>=%g)" % sc["kp_storm_threshold"] if p >= thr
                              else "no major storm",
                "band": _band(p),
                "threshold": thr,
                "operating_point": model["default_operating_point"],
                "winner": model["winner"],
                "test_tss": round(model["metrics"]["tss"], 3),
                "synthetic_model": model.get("synthetic", False),
            }
        except Exception as exc:                               # noqa: BLE001
            notes.append(f"storm model inference failed: {type(exc).__name__}")

    # Climatological fallback probability when the ML track can't run.
    base = model.get("base_rate") if model else None
    fallback_p = base if base is not None else 0.05

    # STEP 6: a forecast auroral oval synthesised from the predicted peak Kp
    # (NOAA's official Kp forecast) so the map's high-latitude layer can switch
    # from "now" (OVATION) to "forecast" severity. Oval synthesis reuses geomag's.
    forecast_kp = noaa.get("max_kp_next_3d") if noaa.get("available") else None
    forecast_oval = None
    if forecast_kp is not None:
        gs = cfg["geomag"].get("grid_step_deg", {}) or {}
        try:
            forecast_oval = geomag.synthesize_oval(forecast_kp, int(gs.get("lat", 2)),
                                                   int(gs.get("lon", 4)))
        except Exception:                                      # noqa: BLE001
            forecast_oval = None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": f"P(Kp>={sc['kp_storm_threshold']:g} within {sc['prediction_window_h']}h)",
        "data_status": status,                                  # live | cached | unavailable
        "solar_wind": summary,                                 # current L1 drivers
        "ml_forecast": ml,                                     # the trained model track
        "noaa_kp_forecast": noaa,                              # official corroboration
        "forecast_kp": forecast_kp,                            # predicted peak Kp (NOAA)
        "forecast_g_scale": geomag.kp_to_g(forecast_kp) if forecast_kp is not None else None,
        "forecast_oval": forecast_oval,                        # synthesised oval at the forecast Kp
        "forecast_aurora_view": geomag.aurora_view(forecast_kp) if forecast_kp is not None else None,
        "headline_p_storm": ml["p_storm"] if ml.get("available") else fallback_p,
        "headline_source": "model" if ml.get("available") else "climatology",
        "notes": notes,
        "disclaimer": "Experimental geomagnetic-storm forecast from the L1 solar "
                      "wind (NASA OMNI-trained). Separate model + feature space from "
                      "the flare forecast. Use alongside official NOAA SWPC products.",
    }


def main():
    ap = argparse.ArgumentParser(description="Train the geomagnetic-storm model.")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on the offline synthetic fixture (no download)")
    args = ap.parse_args()
    train_model(synthetic=args.synthetic)


# late import to avoid a cycle at module load (sources imports config only)
from . import sources  # noqa: E402

if __name__ == "__main__":
    main()
