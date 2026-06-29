"""High-latitude geomagnetic hazard layer — auroral oval + planetary Kp.

This is the OTHER space-weather mechanism, kept deliberately separate from the
dayside HF-radio blackout in hazard.py. Geomagnetic activity is driven by the
solar wind coupling to *Earth's* magnetic field and concentrates at high
*magnetic* latitudes — power-grid GIC, GNSS error, polar-route aviation, aurora.
It is NOT the same feature space as the X-ray-driven flare/blackout layer.

A *visualization of real NOAA products*, not a new prediction:
  - NOAA OVATION gives the current probability (%) of visible aurora on a global
    1-degree grid (downsampled here for the map).
  - The planetary Kp index gives current global geomagnetic activity, mapped to
    NOAA's G-scale (G1..G5).

Output mirrors build_hazard() (freshness / notes / fallback) so the front end
treats it uniformly. Never raises to the caller.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from . import sources
from .config import load_config


def kp_to_g(kp) -> int | None:
    """Map a planetary Kp value to the NOAA geomagnetic-storm G-scale (0-5)."""
    if kp is None:
        return None
    for thresh, g in ((9, 5), (8, 4), (7, 3), (6, 2), (5, 1)):
        if kp >= thresh:
            return g
    return 0


def aurora_view(kp) -> dict:
    """How far toward the equator aurora may be visible, from Kp (STEP 8).

    Equatorward edge of the auroral oval ~ 66.5 - 2*Kp (geomagnetic latitude);
    it can be seen low on the poleward horizon a few degrees further. A documented
    approximation for an at-a-glance readout, not a viewing guarantee."""
    if kp is None:
        return {"available": False}
    overhead = round(66.5 - 2.0 * float(kp), 1)
    horizon = round(overhead - 4.0, 1)
    return {"available": True, "kp": kp, "overhead_mlat": overhead,
            "horizon_mlat": horizon,
            "note": f"At Kp {float(kp):g}, aurora may be overhead near {overhead:g}° "
                    f"magnetic latitude and low on the horizon down to ~{horizon:g}°."}


def _current_kp(cfg: dict) -> dict:
    """Latest planetary Kp + G-scale. Robust to both the object feed
    ({time_tag, Kp, ...}) and the legacy header+rows list-of-lists feed."""
    res = sources.get_kp(cfg)
    data = res.data
    if not res.ok or not isinstance(data, list) or not data:
        return {"kp": None, "g_scale": None, "observed_at": None, "status": "unavailable"}
    kp = obs = None
    first = data[0]
    if isinstance(first, dict):                          # list of objects
        for row in reversed(data):
            try:
                kp = float(row.get("Kp")); obs = row.get("time_tag"); break
            except (TypeError, ValueError):
                continue
    elif isinstance(first, list):                        # header row + data rows
        header = [str(h).lower() for h in first]
        kpi = header.index("kp") if "kp" in header else 1
        ti = header.index("time_tag") if "time_tag" in header else 0
        for row in reversed(data[1:]):
            try:
                kp = float(row[kpi]); obs = row[ti]; break
            except (TypeError, ValueError, IndexError):
                continue
    return {"kp": kp, "g_scale": kp_to_g(kp), "observed_at": obs, "status": res.status}


def parse_ovation(obj: dict, step_lat: int = 2, step_lon: int = 4) -> dict | None:
    """OVATION coordinates ([lon 0-359, lat -90..90, aurora %]) -> a downsampled
    {lats(+90..-90), lons(-180..180), values[%]} grid in the SAME shape as the
    D-RAP hazard grid, so the front end can texture it with the same code path."""
    coords = obj.get("coordinates") if isinstance(obj, dict) else None
    if not coords:
        return None
    full = [[0.0] * 360 for _ in range(181)]             # full[lat+90][lon 0..359]
    for c in coords:
        try:
            lon = int(round(float(c[0]))) % 360
            lat = int(round(float(c[1])))
            val = float(c[2])
        except (TypeError, ValueError, IndexError):
            continue
        if -90 <= lat <= 90:
            full[lat + 90][lon] = val
    lats = list(range(90, -91, -step_lat))
    lons = list(range(-180, 180, step_lon))
    values = [[full[la + 90][lo % 360] for lo in lons] for la in lats]
    return {"lats": lats, "lons": lons, "values": values,
            "observation_time": obj.get("Observation Time"),
            "forecast_time": obj.get("Forecast Time")}


def synthesize_oval(kp, step_lat: int = 2, step_lon: int = 4) -> dict:
    """Fallback when OVATION is down: a symmetric auroral band whose equatorward
    edge moves toward the equator as Kp rises (~66.5 - 2*Kp deg). Documented
    approximation by GEOGRAPHIC latitude (real ovals follow magnetic latitude)."""
    k = kp if kp is not None else 3.0
    boundary = 66.5 - 2.0 * k                            # equatorward edge (deg)
    center = boundary + 4.0
    lats = list(range(90, -91, -step_lat))
    lons = list(range(-180, 180, step_lon))
    values = []
    for la in lats:
        al = abs(la)
        v = round(80.0 * math.exp(-((al - center) ** 2) / (2 * 6.0 ** 2)), 1) if al >= boundary - 6 else 0.0
        values.append([v] * len(lons))                   # zonally uniform (rough)
    return {"lats": lats, "lons": lons, "values": values,
            "observation_time": None, "forecast_time": None}


def classify_aurora_cells(grid: dict, cfg: dict, cap: int = 400) -> list[dict]:
    g = cfg["geomag"]
    watch, warn, severe = g["aurora_watch_pct"], g["aurora_warning_pct"], g["aurora_severe_pct"]
    cells = []
    for la, row in zip(grid["lats"], grid["values"]):
        for lo, v in zip(grid["lons"], row):
            if v < watch:
                continue
            level = "severe" if v >= severe else "warning" if v >= warn else "watch"
            cells.append({"lat": la, "lon": lo, "aurora_pct": round(v, 1), "level": level})
    cells.sort(key=lambda c: c["aurora_pct"], reverse=True)
    return cells[:cap]


def build_geomag(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    notes: list[str] = []
    step = cfg["geomag"].get("grid_step_deg", {}) or {}
    slat, slon = int(step.get("lat", 2)), int(step.get("lon", 4))

    kp = _current_kp(cfg)

    ov = sources.get_ovation(cfg)
    grid, source, status = None, None, "unavailable"
    if ov.ok:
        grid = parse_ovation(ov.data, slat, slon)
        if grid:
            source, status = "noaa-ovation", ov.status
            notes.extend(ov.notes)
        else:
            notes.append("OVATION fetched but could not be parsed.")
    if grid is None:
        grid = synthesize_oval(kp.get("kp"), slat, slon)
        source, status = "synthesized-from-kp", "fallback"
        notes.append("OVATION unavailable — auroral oval synthesised from current "
                     "Kp (approximation, geographic-latitude band).")

    danger = classify_aurora_cells(grid, cfg)
    peak = max((c["aurora_pct"] for c in danger),
               default=round(max((max(r) for r in grid["values"]), default=0.0), 1))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kp": kp,                                        # {kp, g_scale, observed_at, status}
        "aurora_view": aurora_view(kp.get("kp")),        # how far south the aurora may be visible

        "aurora": {
            "source": source,                            # noaa-ovation | synthesized-from-kp
            "status": status,                            # live | cached | fallback
            "observation_time": grid.get("observation_time"),
            "forecast_time": grid.get("forecast_time"),
            "units": "% chance of visible aurora",
            "resolution_deg": {"lat": slat, "lon": slon},
            "peak_pct": peak,
            "grid": grid,                                # downsampled field, for the map texture
        },
        "danger_cells": danger,                          # cells above the watch threshold (capped)
        "thresholds_pct": {
            "watch": cfg["geomag"]["aurora_watch_pct"],
            "warning": cfg["geomag"]["aurora_warning_pct"],
            "severe": cfg["geomag"]["aurora_severe_pct"],
        },
        "kp_thresholds": {
            "watch": cfg["geomag"]["kp_watch"],
            "warning": cfg["geomag"]["kp_warning"],
            "severe": cfg["geomag"]["kp_severe"],
        },
        "disclaimer": "High-latitude geomagnetic/auroral hazard (NOAA OVATION + "
                      "planetary Kp). A separate mechanism from the dayside HF-radio "
                      "blackout: it affects power grids, GNSS and polar aviation at "
                      "high magnetic latitudes. Visualization of NOAA products, not a "
                      "new prediction.",
        "notes": notes,
    }
