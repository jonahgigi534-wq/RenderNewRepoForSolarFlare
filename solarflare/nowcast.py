"""Nowcast = what is happening RIGHT NOW, computed exactly from GOES flux.

Because a flare's class is *defined* by its 1-8A peak flux, the current state
needs no ML — it is a measurement. This module:
  * reports the current flux and its class,
  * finds the peak flare in the last N hours,
  * raises the escalating X-class warning,
  * extracts recent-activity features the statistical forecaster consumes.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from . import labels
from .config import load_config


def _to_series(rows):
    times, flux = [], []
    for d in rows:
        f = d.get("flux")
        if f is None:
            continue
        times.append(datetime.fromisoformat(d["time_tag"].replace("Z", "+00:00")))
        flux.append(float(f))
    return times, np.asarray(flux, dtype=float)


def nowcast(rows, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    times, flux = _to_series(rows)
    if flux.size == 0:
        return {"available": False, "reason": "no flux samples"}

    now_flux = float(flux[-1])
    now_class = labels.flux_to_letter(now_flux)
    now_cat = labels.flux_to_category(now_flux, cfg)

    # Peak over the last 24h.
    cutoff = times[-1].timestamp() - 24 * 3600
    recent_mask = np.array([t.timestamp() >= cutoff for t in times])
    peak24 = float(flux[recent_mask].max()) if recent_mask.any() else now_flux
    peak24_class = labels.flux_to_letter(peak24)

    warn = labels.x_warning(max(now_flux, peak24), cfg)

    return {
        "available": True,
        "observed_at": times[-1].astimezone(timezone.utc).isoformat(),
        "current_flux_wm2": now_flux,
        "current_class": now_class,
        "current_category": now_cat,
        "peak_24h_flux_wm2": peak24,
        "peak_24h_class": peak24_class,
        "is_flaring": now_cat != "no-flare",
        "x_warning": {"level": warn.level, "message": warn.label, "scale": warn.code},
    }


def _count_excursions(seg: np.ndarray, threshold: float) -> int:
    """Number of DISTINCT flares above `threshold` in a flux segment.

    A flare is one maximal run of samples >= threshold (a rising threshold
    crossing), NOT every individual 1-minute sample. Counting samples would
    turn a single multi-hour enhancement into hundreds of phantom events.
    """
    above = seg >= threshold
    if above.size == 0:
        return 0
    # Rising edges: False -> True transitions, plus the first sample if already high.
    rising = np.sum(above[1:] & ~above[:-1])
    return int(rising + (1 if above[0] else 0))


def activity_features(rows, cfg: dict | None = None) -> dict:
    """Summarise recent flux for the statistical forecaster (24h & 48h windows)."""
    cfg = cfg or load_config()
    times, flux = _to_series(rows)
    if flux.size == 0:
        return {}
    ts = np.array([t.timestamp() for t in times])
    out = {}
    for hrs in (6, 12, 24, 48):
        m = ts >= (ts[-1] - hrs * 3600)
        seg = flux[m]
        if seg.size == 0:
            continue
        out[f"max_{hrs}h"] = float(seg.max())
        out[f"mean_{hrs}h"] = float(seg.mean())
        # Distinct C+ and M+ flares in the window (proxy for AR productivity).
        out[f"n_C_{hrs}h"] = _count_excursions(seg, 1e-6)
        out[f"n_M_{hrs}h"] = _count_excursions(seg, 1e-5)
    # Short-term trend: last 6h mean vs prior 6h mean.
    last6 = flux[ts >= ts[-1] - 6 * 3600]
    prev6 = flux[(ts < ts[-1] - 6 * 3600) & (ts >= ts[-1] - 12 * 3600)]
    if last6.size and prev6.size:
        out["trend_ratio"] = float(last6.mean() / max(prev6.mean(), 1e-12))
    return out
