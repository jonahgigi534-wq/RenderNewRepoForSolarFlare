"""Live SHARP flare inference.

Fetches the most recent 12 h of JSOC SHARP data for each on-disk active region
and runs OUR saved pipeline (impute -> standardize -> classifier) to get
P(M-class+ flare within 24 h) per region and full-disk. The transform is
identical to training (same keywords, same 60-record window, same summary-stat
features, same persisted pipeline) — that's the whole point of owning it.

Live fetches use config `sharp_live.live_series` (hmi.sharp_cea_720s_nrt, ~1 h
fresh) because the definitive series lags real time by weeks; training and
dataset builds stay on the definitive series (see sharpdata.py).

Resilient: never raises to the API; degrades to {"available": False, ...} when the
model is missing, `drms` is unavailable, or JSOC returns nothing.

Pass `at_time` (a historical UTC datetime) for a historical demo; `at_time=None`
uses 'now'.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import data as dataio
from .config import load_config

_MODELS: dict = {}          # variant key ("" = default) -> loaded payload or None
_CACHE: dict = {}           # (variant, at_key) -> (timestamp, result)
_WINDOWS_CACHE: dict = {}   # at_key -> (timestamp, windows) — shared across variants


def variant_spec(cfg: dict, variant: str | None):
    """(label, model_path) for a variant key, or the default when None/''."""
    sl = cfg["sharp_live"]
    if not variant:
        return "Deployed (SWAN-SF-trained)", sl.get("model_path", "models/flare_sharp_live_model.joblib")
    spec = sl.get("variants", {}).get(variant)
    if spec is None:
        return None, None
    return spec.get("label", variant), spec["model_path"]


def list_variants(cfg: dict) -> list[dict]:
    """[{key, label, trained}] for the dashboard's model selector — the default
    deployed model first, then every configured variant that has been trained."""
    out = [{"key": "", "label": variant_spec(cfg, None)[0],
           "trained": load_model(cfg, None) is not None}]
    for key, spec in cfg["sharp_live"].get("variants", {}).items():
        out.append({"key": key, "label": spec.get("label", key),
                    "trained": load_model(cfg, key) is not None})
    return out


def load_model(cfg: dict, variant: str | None = None):
    """Load + cache the saved live SHARP pipeline for a variant ("" / None =
    the default deployed model), or None if not trained / unknown variant."""
    key = variant or ""
    if key in _MODELS:
        return _MODELS[key]
    _, path = variant_spec(cfg, variant)
    if path is None:
        _MODELS[key] = None
        return None
    if not os.path.isabs(path):                           # resolve against project root, not CWD
        path = os.path.join(cfg.get("_project_root", "."), path)
    if not os.path.exists(path):
        _MODELS[key] = None
        return None
    try:
        import joblib
        _MODELS[key] = joblib.load(path)
    except Exception:                                     # noqa: BLE001
        _MODELS[key] = None
    return _MODELS[key]


def fetch_recent_windows(cfg: dict, at_time: datetime | None = None):
    """Per region with a full recent window: (harpnum, noaa_ar, end_time, X(60,17))."""
    from . import sharpdata
    sl = cfg["sharp_live"]
    obs_h = int(sl["observation_window_h"])
    at = at_time or datetime.now(timezone.utc)
    t0 = at - timedelta(hours=obs_h * 1.4)                # a little slack to ensure 60 records
    # Live ("now") queries use the near-real-time series; historical demos use
    # the definitive series, which is complete for the past.
    series = (sl.get("live_series") or sharpdata.SERIES) if at_time is None else sharpdata.SERIES
    rows = sharpdata.fetch_sharp(t0, at, cfg, chunk_days=2, verbose=False, series=series)
    win = obs_h * 60 // sharpdata.CADENCE_MIN
    by_harp: dict[int, list] = {}
    for t, harp, ar, vec in rows:
        by_harp.setdefault(harp, []).append((t, ar, vec))
    out = []
    for harp, recs in by_harp.items():
        recs.sort(key=lambda r: r[0])
        if len(recs) < win:
            continue
        block = recs[-win:]                               # the most recent 12 h
        span_h = (block[-1][0] - block[0][0]).total_seconds() / 3600.0
        if span_h > obs_h * 1.5:
            continue
        mat = np.stack([b[2] for b in block])
        if np.isfinite(mat).mean() < 0.7:
            continue
        ars = [b[1] for b in block if b[1]]
        ar = max(set(ars), key=ars.count) if ars else 0
        out.append((harp, ar, block[-1][0], mat))
    return out


def _cached_windows(cfg: dict, at_time: datetime | None, at_key: str, use_cache: bool):
    """JSOC windows shared across variants — comparing two live models should
    not double the JSOC hits, since both consume the identical recent data."""
    ttl = float(cfg["sharp_live"].get("cache_minutes", 15)) * 60
    if use_cache:
        hit = _WINDOWS_CACHE.get(at_key)
        if hit and (time.time() - hit[0]) < ttl:
            return hit[1]
    windows = fetch_recent_windows(cfg, at_time)
    if use_cache:
        _WINDOWS_CACHE[at_key] = (time.time(), windows)
    return windows


def predict_live(cfg: dict | None = None, at_time: datetime | None = None,
                 use_cache: bool = True, variant: str | None = None) -> dict:
    """Cached wrapper: a live result is reused for cache_minutes so repeated
    dashboard polls don't re-hit JSOC each time. `variant` selects an alternate
    deployable model from config sharp_live.variants (default: the deployed
    model). Never raises."""
    cfg = cfg or load_config()
    at_key = at_time.isoformat() if at_time else "now"
    cache_key = (variant or "", at_key)
    ttl = float(cfg["sharp_live"].get("cache_minutes", 15)) * 60
    if use_cache:
        hit = _CACHE.get(cache_key)
        if hit and (time.time() - hit[0]) < ttl:
            return hit[1]
    res = _predict_live_impl(cfg, at_time, at_key, variant, use_cache)
    if use_cache:
        _CACHE[cache_key] = (time.time(), res)
    return res


def _predict_live_impl(cfg: dict, at_time: datetime | None, at_key: str,
                       variant: str | None, use_cache: bool) -> dict:
    label, _ = variant_spec(cfg, variant)
    if label is None:
        known = sorted(cfg["sharp_live"].get("variants", {}))
        return {"available": False, "reason": f"unknown variant (known: {known})"}
    model = load_model(cfg, variant)
    if model is None:
        return {"available": False, "reason": f"'{label}' model not trained yet",
                "variant": variant or "", "variant_label": label}
    info = {"model": model.get("winner"), "variant": variant or "", "variant_label": label,
            "test_tss": round(model.get("metrics", {}).get("tss", 0.0), 3),
            "trained_on": model.get("data_span")}
    from . import sharpdata
    series = (cfg["sharp_live"].get("live_series") or sharpdata.SERIES) \
        if at_time is None else sharpdata.SERIES
    try:
        windows = _cached_windows(cfg, at_time, at_key, use_cache)
    except Exception as exc:                              # noqa: BLE001 (never raise)
        return {"available": False, "reason": f"JSOC fetch unavailable ({type(exc).__name__})", **info}
    if not windows:
        return {"available": False, **info,
                "reason": f"no recent JSOC data on {series} for this window"}

    X3d = np.stack([w[3] for w in windows])               # (k, 60, 17)
    Xf = dataio.build_matrix(X3d, cfg)                    # (k, 119) — same features as training
    proba = model["model"].predict_proba(Xf)[:, 1]
    # Operating point is config-selectable; "operational" exists only after
    # `python -m solarflare.recalibrate` saved a live-data-recalibrated
    # threshold into the artifact (the self-correcting-deployment feature).
    op_name = str(cfg["sharp_live"].get("operating_point", "balanced"))
    ops = model.get("operating_points", {}) or {}
    if op_name in ops:
        thr = float(ops[op_name]["threshold"])
    else:                                                 # fall back to the training default
        op_name = str(model.get("default_operating_point", "balanced"))
        thr = float(model["threshold"])
    recal_year = (ops.get("operational", {}) or {}).get("calibrated_on_year") \
        if op_name == "operational" else None
    regions = []
    for (harp, ar, end_t, _), p in zip(windows, proba):
        regions.append({"harpnum": int(harp), "noaa_ar": int(ar),
                        "p_M_or_greater_24h": round(float(p), 4),
                        "will_flare": bool(p >= thr)})
    regions.sort(key=lambda r: r["p_M_or_greater_24h"], reverse=True)
    full = 1.0 - float(np.prod(1.0 - proba))              # P(any region flares)
    latest = max(w[2] for w in windows)                   # newest T_REC actually used
    age_min = ((at_time or datetime.now(timezone.utc)) - latest).total_seconds() / 60.0
    return {
        "available": True,
        "model": model.get("winner"),
        "variant": variant or "",
        "variant_label": label,
        "test_tss": round(model.get("metrics", {}).get("tss", 0.0), 3),
        "as_of": (at_time or datetime.now(timezone.utc)).isoformat(),
        "data_series": series,
        "latest_data": latest.isoformat(),
        "data_age_minutes": round(max(0.0, age_min), 1),
        "operating_point": op_name,
        "recalibrated_on_year": recal_year,   # set when the operational (self-corrected) point is active
        "threshold": thr,
        "p_M_or_greater_24h_fulldisk": round(full, 4),
        "n_regions": len(regions),
        "regions": regions,
        "disclaimer": "Live SHARP ML — P(M+ flare in 24h) per active region from JSOC "
                      "magnetic-field data, using our own JSOC-trained model (saved "
                      "preprocessing). Real ML; separate from the SWAN-SF benchmark model.",
    }
