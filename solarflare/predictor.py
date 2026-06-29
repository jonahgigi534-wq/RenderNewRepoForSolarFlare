"""The predictor: fetch live data -> nowcast + forecasts -> one clean result.

Fallback chain (always returns something honest):
  1. Fetch GOES flux. If live -> full nowcast + flux forecast.
  2. If flux fetch failed but cache exists -> same, flagged "stale".
  3. If everything is down -> climatological fallback forecast + clear notice.
  4. SHARP ML model is ensembled in whenever a magnetic feature vector is
     supplied (via predict(... sharp_features=...)); otherwise the flux track
     stands alone. The model is NEVER required for the API to respond.

The output dict is JSON-ready and is what both the CLI and the web API return.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np

from . import fluxmodel, labels, nowcast, sources
from .config import load_config


# ----------------------------------------------------------------------
# SHARP model loading (optional, lazy, fault-tolerant)
# ----------------------------------------------------------------------
_MODEL_CACHE: dict | None = None


def load_sharp_model(cfg: dict | None = None):
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    cfg = cfg or load_config()
    path = os.path.join(cfg["paths"]["model_dir"], "flare_sharp_model.joblib")
    if not os.path.exists(path):
        return None
    try:
        import joblib
        _MODEL_CACHE = joblib.load(path)
    except Exception:                          # noqa: BLE001
        _MODEL_CACHE = None
    return _MODEL_CACHE


def sharp_forecast(sharp_features, cfg: dict | None = None) -> dict | None:
    """Run the SWAN-SF model on a single engineered feature vector (1, n_feat).

    Decision uses the configured operating point (live.operating_point), so the
    same probability can be read as high-recall, balanced, or high-precision.
    """
    cfg = cfg or load_config()
    payload = load_sharp_model(cfg)
    if payload is None or sharp_features is None:
        return None
    X = np.asarray(sharp_features, dtype=float).reshape(1, -1)
    proba = float(payload["model"].predict_proba(X)[0, 1])
    op_name = cfg["live"]["operating_point"]
    ops = payload.get("operating_points", {})
    threshold = ops.get(op_name, {}).get("threshold", payload.get("threshold", 0.5))
    return {
        "available": True,
        "model": f"swansf-sharp-{payload.get('winner', 'model')}",
        "trained_at": payload.get("trained_at"),
        "test_tss": payload.get("metrics", {}).get("tss"),
        "p_M_or_greater_24h": round(proba, 4),
        "band": labels.probability_band(proba, cfg)["name"],
        "operating_point": op_name,
        "threshold": round(threshold, 3),
        "prediction": "M+ flare likely" if proba >= threshold else "no significant flare",
    }


# ----------------------------------------------------------------------
# Ensemble
# ----------------------------------------------------------------------
def _ensemble_24h(flux_fc: dict, sharp_fc: dict | None, noaa_fc: dict | None) -> dict:
    """Blend the flux, SHARP and NOAA-official M+ probabilities (24h)."""
    p_flux = flux_fc["horizons"]["24h"]["p_M_or_greater"] if flux_fc.get("available") else None
    p_sharp = sharp_fc["p_M_or_greater_24h"] if sharp_fc else None
    p_noaa = noaa_fc.get("p_M_or_greater_24h") if noaa_fc and noaa_fc.get("available") else None
    # Weights reflect trust: NOAA's operational forecast and the learned SHARP
    # model outrank the simple flux-persistence baseline.
    parts, weights, used = [], [], []
    for p, w, name in ((p_noaa, 0.45, "noaa"), (p_sharp, 0.35, "sharp"), (p_flux, 0.20, "flux")):
        if p is not None:
            parts.append(p); weights.append(w); used.append(name)
    if not parts:
        return {"p_M_or_greater_24h": None, "sources": []}
    p = float(np.average(parts, weights=weights))
    return {"p_M_or_greater_24h": round(p, 4), "sources": used}


def _climatology_fallback() -> dict:
    """Last-resort forecast when all live data is unavailable (long-run base rates)."""
    base = {  # rough solar-cycle-averaged daily probabilities
        "12h": {"p_C_or_greater": 0.35, "p_M_or_greater": 0.08, "p_X_class": 0.008},
        "24h": {"p_C_or_greater": 0.55, "p_M_or_greater": 0.15, "p_X_class": 0.015},
        "48h": {"p_C_or_greater": 0.75, "p_M_or_greater": 0.27, "p_X_class": 0.03},
    }
    for h in base.values():
        h["expected_max_class"] = "C-M"
    return {"available": True, "model": "climatology-fallback", "horizons": base}


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def predict(sharp_features=None, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    notes: list[str] = []

    flux_res = sources.get_xray_flux(cfg)
    if flux_res.ok:
        rows = flux_res.data
        now = nowcast.nowcast(rows, cfg)
        activity = nowcast.activity_features(rows, cfg)
        flux_fc = fluxmodel.forecast(activity, cfg)
        data_status = flux_res.status
        notes.extend(flux_res.notes)
    else:
        now = {"available": False, "reason": flux_res.error}
        flux_fc = _climatology_fallback()
        data_status = "unavailable"
        notes.append("Live X-ray data unavailable — using climatological fallback.")
        notes.extend(flux_res.notes)

    sharp_fc = sharp_forecast(sharp_features, cfg) if sharp_features is not None else None
    if sharp_features is not None and sharp_fc is None:
        notes.append("SHARP model not found/loadable — flux track only. "
                     "Run `python -m solarflare.train` to enable it.")

    # Trained-model provenance (shown even when the SHARP track isn't run live).
    payload = load_sharp_model(cfg)
    sharp_info = None
    if payload is not None:
        m = payload.get("metrics", {})
        op_name = cfg["live"]["operating_point"]
        ops_summary = {}
        for name, op in payload.get("operating_points", {}).items():
            t = op.get("test", {})
            ops_summary[name] = {
                "threshold": op.get("threshold"),
                "tss": round(t.get("tss", 0), 3),
                "recall": round(t.get("recall", 0), 3),
                "precision": round(t.get("precision", 0), 3),
            }
        sharp_info = {
            "trained": True,
            "winner": payload.get("winner"),
            "trained_at": payload.get("trained_at"),
            "synthetic": payload.get("synthetic"),
            "test_tss": m.get("tss"),
            "test_recall": m.get("recall"),
            "test_precision": m.get("precision"),
            "test_samples": m.get("n"),
            "operating_point": op_name,
            "operating_points": ops_summary,
            "running_live": sharp_fc is not None,
        }

    # NOAA's official per-region forecast (best-effort, never blocks).
    noaa_fc = sources.noaa_region_forecast(cfg)
    n_regions = noaa_fc.get("n_regions") if noaa_fc.get("available") else None

    ensemble = _ensemble_24h(flux_fc, sharp_fc, noaa_fc)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_freshness": {
            "status": data_status,                 # live | cached | unavailable
            "xray_age_minutes": round(flux_res.age_minutes, 1) if flux_res.age_minutes is not None else None,
            "source": flux_res.source,
        },
        "nowcast": now,
        "forecast": {
            "flux_track": flux_fc,
            "sharp_track": sharp_fc,
            "sharp_model": sharp_info,
            "noaa_track": noaa_fc,
            "ensemble_24h": ensemble,
        },
        "active_region_count": n_regions,
        "notes": notes,
        "disclaimer": "Probabilistic guidance, not a guarantee. Flare forecasting "
                      "skill is inherently limited; use alongside official NOAA SWPC products.",
    }


def main():
    import json
    print(json.dumps(predict(), indent=2, default=str))


if __name__ == "__main__":
    main()
