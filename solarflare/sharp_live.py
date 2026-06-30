"""Live SHARP flare inference.

Fetches the most recent 12 h of JSOC `hmi.sharp_cea_720s` data for each on-disk
active region and runs OUR saved pipeline (impute -> standardize -> classifier)
to get P(M-class+ flare within 24 h) per region and full-disk. The transform is
identical to training (same keywords, same 60-record window, same summary-stat
features, same persisted pipeline) — that's the whole point of owning it.

Resilient: never raises to the API; degrades to {"available": False, ...} when the
model is missing, `drms` is unavailable, or JSOC returns nothing.

NOTE: in a sandbox whose clock is past real JSOC data, pass `at_time` (a historical
UTC datetime) to demo. On a real machine, `at_time=None` uses 'now'.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import data as dataio
from .config import load_config

_MODEL = None
_CACHE: dict = {}


def load_model(cfg: dict):
    """Load + cache the saved live SHARP pipeline, or None if not trained yet."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = cfg["sharp_live"].get("model_path", "models/flare_sharp_live_model.joblib")
    if not os.path.isabs(path):                           # resolve against project root, not CWD
        path = os.path.join(cfg.get("_project_root", "."), path)
    if not os.path.exists(path):
        return None
    try:
        import joblib
        _MODEL = joblib.load(path)
    except Exception:                                     # noqa: BLE001
        _MODEL = None
    return _MODEL


def fetch_recent_windows(cfg: dict, at_time: datetime | None = None):
    """Per region with a full recent window: (harpnum, noaa_ar, end_time, X(60,17))."""
    from . import sharpdata
    sl = cfg["sharp_live"]
    obs_h = int(sl["observation_window_h"])
    at = at_time or datetime.now(timezone.utc)
    t0 = at - timedelta(hours=obs_h * 1.4)                # a little slack to ensure 60 records
    rows = sharpdata.fetch_sharp(t0, at, cfg, chunk_days=2, verbose=False)
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


def predict_live(cfg: dict | None = None, at_time: datetime | None = None,
                 use_cache: bool = True) -> dict:
    """Cached wrapper: a live result is reused for cache_minutes so repeated
    dashboard polls don't re-hit JSOC each time. Never raises."""
    cfg = cfg or load_config()
    key = at_time.isoformat() if at_time else "now"
    ttl = float(cfg["sharp_live"].get("cache_minutes", 15)) * 60
    if use_cache:
        hit = _CACHE.get(key)
        if hit and (time.time() - hit[0]) < ttl:
            return hit[1]
    res = _predict_live_impl(cfg, at_time)
    if use_cache:
        _CACHE[key] = (time.time(), res)
    return res


def _predict_live_impl(cfg: dict, at_time: datetime | None) -> dict:
    model = load_model(cfg)
    if model is None:
        return {"available": False, "reason": "live SHARP model not trained yet"}
    info = {"model": model.get("winner"),
            "test_tss": round(model.get("metrics", {}).get("tss", 0.0), 3),
            "trained_on": model.get("data_span")}
    try:
        windows = fetch_recent_windows(cfg, at_time)
    except Exception as exc:                              # noqa: BLE001 (never raise)
        return {"available": False, "reason": f"JSOC fetch unavailable ({type(exc).__name__})", **info}
    if not windows:
        return {"available": False, **info,
                "reason": "no current JSOC data in this environment "
                          "(the model is trained and runs live on a real machine)"}

    X3d = np.stack([w[3] for w in windows])               # (k, 60, 17)
    Xf = dataio.build_matrix(X3d, cfg)                    # (k, 119) — same features as training
    proba = model["model"].predict_proba(Xf)[:, 1]
    thr = float(model["threshold"])
    regions = []
    for (harp, ar, end_t, _), p in zip(windows, proba):
        regions.append({"harpnum": int(harp), "noaa_ar": int(ar),
                        "p_M_or_greater_24h": round(float(p), 4),
                        "will_flare": bool(p >= thr)})
    regions.sort(key=lambda r: r["p_M_or_greater_24h"], reverse=True)
    full = 1.0 - float(np.prod(1.0 - proba))              # P(any region flares)
    return {
        "available": True,
        "model": model.get("winner"),
        "test_tss": round(model.get("metrics", {}).get("tss", 0.0), 3),
        "as_of": (at_time or datetime.now(timezone.utc)).isoformat(),
        "operating_point": model.get("default_operating_point"),
        "threshold": thr,
        "p_M_or_greater_24h_fulldisk": round(full, 4),
        "n_regions": len(regions),
        "regions": regions,
        "disclaimer": "Live SHARP ML — P(M+ flare in 24h) per active region from JSOC "
                      "magnetic-field data, using our own JSOC-trained model (saved "
                      "preprocessing). Real ML; separate from the SWAN-SF benchmark model.",
    }
