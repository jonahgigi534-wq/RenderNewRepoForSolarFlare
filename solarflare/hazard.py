"""Geographic hazard layer — dayside HF-radio-blackout footprint.

This is a *visualization of real NOAA products*, not a new prediction model:
NOAA's D-RAP gives the global grid of the highest radio frequency (MHz) being
absorbed in the ionosphere right now. Solar X-rays only hit the sunlit side, so
the blackout footprint sits under the subsolar point — which we compute from
pure astronomy. Same X-ray physics that drives the flare model; kept entirely
separate from the (later) high-latitude geomagnetic hazard.

Output of `build_hazard()` mirrors the other endpoints (freshness/notes/fallback)
so the front end can treat it uniformly.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from . import nowcast, sources
from .config import load_config


# ----------------------------------------------------------------------
# Astronomy: subsolar point + solar zenith (no network needed)
# ----------------------------------------------------------------------
def subsolar_point(dt: datetime | None = None) -> tuple[float, float]:
    """Latitude/longitude where the Sun is directly overhead (NOAA SPA formulas)."""
    dt = dt or datetime.now(timezone.utc)
    doy = int(dt.strftime("%j"))
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    gamma = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma) - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))                  # minutes
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))  # rad
    sub_lat = math.degrees(decl)
    utc_min = dt.hour * 60 + dt.minute + dt.second / 60
    sub_lon = (720 - utc_min - eqtime) / 4.0
    sub_lon = ((sub_lon + 180) % 360) - 180          # normalise to [-180, 180]
    return round(sub_lat, 3), round(sub_lon, 3)


def solar_zenith_cos(lat, lon, sub_lat, sub_lon) -> float:
    """cos(solar zenith angle); > 0 means the Sun is above the horizon (dayside)."""
    la, lo = math.radians(lat), math.radians(lon)
    sla, slo = math.radians(sub_lat), math.radians(sub_lon)
    return (math.sin(la) * math.sin(sla)
            + math.cos(la) * math.cos(sla) * math.cos(lo - slo))


# ----------------------------------------------------------------------
# D-RAP grid parsing
# ----------------------------------------------------------------------
def parse_drap(text: str) -> dict | None:
    """Parse drap_global_frequencies.txt into {lats, lons, values, valid_at}.

    values[i][j] = highest affected frequency (MHz) at lats[i], lons[j].
    Grid resolution is read from the file (2 deg lat x 4 deg lon), not assumed.
    """
    lons, lats, rows, valid_at = None, [], [], None
    for line in text.splitlines():
        if "Product Valid At" in line:
            valid_at = line.split(":", 1)[1].strip()
            continue
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if set(s) <= set("-"):                       # dashed separator
            continue
        if "|" in line:
            left, right = line.split("|", 1)
            try:
                lat = float(left.strip())
                vals = [float(x) for x in right.split()]
            except ValueError:
                continue
            lats.append(lat)
            rows.append(vals)
        else:                                        # longitude axis line
            try:
                nums = [float(x) for x in s.split()]
            except ValueError:
                continue
            if len(nums) > 10:
                lons = nums
    if not lons or not rows:
        return None
    width = len(lons)
    rows = [r for r in rows if len(r) == width]      # guard ragged rows
    if not rows:
        return None
    return {"lats": lats[:len(rows)], "lons": lons, "values": rows,
            "valid_at": valid_at}


# ----------------------------------------------------------------------
# Fallback: synthesise a dayside field from the current flare class
# ----------------------------------------------------------------------
def _flare_peak_haf_mhz(flux: float) -> float:
    """Rough peak highest-affected-frequency (MHz) at the subsolar point for a
    given GOES flux. Calibrated to NOAA R-scale behaviour: ~0 below C5, ~10 at
    M1 (R1), ~22 at X1 (R3), ~34 at X10 (R4). Documented approximation used
    ONLY when D-RAP is unavailable."""
    if not flux or flux <= 0:
        return 0.0
    return max(0.0, 12.0 * (math.log10(flux) + 5.0) + 10.0)   # M1->10, X1->22, X10->34


def synthesize_field(sub_lat, sub_lon, flux, step_lat=2, step_lon=4) -> dict:
    """Build a D-RAP-shaped field from the flare class (sunlit side only)."""
    peak = _flare_peak_haf_mhz(flux)
    lats = list(range(89, -90, -step_lat))
    lons = list(range(-178, 180, step_lon))
    values = []
    for la in lats:
        row = []
        for lo in lons:
            cz = solar_zenith_cos(la, lo, sub_lat, sub_lon)
            row.append(round(max(0.0, peak * cz), 1) if cz > 0 else 0.0)
        values.append(row)
    return {"lats": lats, "lons": lons, "values": values, "valid_at": None}


# ----------------------------------------------------------------------
# Danger-cell classification
# ----------------------------------------------------------------------
def classify_cells(grid: dict, cfg: dict) -> list[dict]:
    h = cfg["hazard"]
    watch, warn, severe = h["haf_watch_mhz"], h["haf_warning_mhz"], h["haf_severe_mhz"]
    cells = []
    for la, row in zip(grid["lats"], grid["values"]):
        for lo, v in zip(grid["lons"], row):
            if v < watch:
                continue
            level = "severe" if v >= severe else "warning" if v >= warn else "watch"
            cells.append({"lat": la, "lon": lo, "haf_mhz": round(v, 1), "level": level})
    cells.sort(key=lambda c: c["haf_mhz"], reverse=True)
    return cells


# ----------------------------------------------------------------------
# Public builder
# ----------------------------------------------------------------------
def _current_r_scale(cfg) -> dict:
    res = sources.get_noaa_scales(cfg)
    if res.ok and isinstance(res.data, dict):
        cur = res.data.get("0", {}).get("R", {})
        return {"scale": cur.get("Scale"), "text": cur.get("Text"),
                "status": res.status}
    return {"scale": None, "text": None, "status": "unavailable"}


def build_hazard(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    notes: list[str] = []
    sub_lat, sub_lon = subsolar_point()

    # Current flare context (reuses the existing X-ray/nowcast path).
    flux_res = sources.get_xray_flux(cfg)
    now = nowcast.nowcast(flux_res.data, cfg) if flux_res.ok else {"available": False}
    cur_flux = now.get("current_flux_wm2") if now.get("available") else None
    cur_class = now.get("current_class") if now.get("available") else None

    # D-RAP grid (resilient); fall back to a synthesised dayside field.
    drap = sources.get_drap(cfg)
    grid, source, status = None, None, "unavailable"
    if drap.ok:
        grid = parse_drap(drap.data)
        if grid:
            source, status = "noaa-drap", drap.status
            notes.extend(drap.notes)
        else:
            notes.append("D-RAP fetched but could not be parsed.")
    if grid is None:
        grid = synthesize_field(sub_lat, sub_lon, cur_flux or 0.0)
        source = "synthesized-from-flare-class"
        status = "fallback"
        notes.append("D-RAP unavailable — dayside footprint synthesised from the "
                     "live flare class (approximation, sunlit side only).")

    danger = classify_cells(grid, cfg)
    r_scale = _current_r_scale(cfg)
    peak = max((c["haf_mhz"] for c in danger), default=round(
        max((max(r) for r in grid["values"] if r), default=0.0), 1))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subsolar": {"lat": sub_lat, "lon": sub_lon},
        "hf_blackout": {
            "source": source,                       # noaa-drap | synthesized-from-flare-class
            "status": status,                       # live | cached | fallback
            "valid_at": grid.get("valid_at"),
            "units": "MHz (highest affected frequency)",
            "resolution_deg": {
                "lat": abs(grid["lats"][1] - grid["lats"][0]) if len(grid["lats"]) > 1 else None,
                "lon": abs(grid["lons"][1] - grid["lons"][0]) if len(grid["lons"]) > 1 else None,
            },
            "peak_haf_mhz": peak,
            "current_flare_class": cur_class,
            "current_flux_wm2": cur_flux,
            "r_scale": r_scale,                     # NOAA radio-blackout scale (0-5)
            "grid": grid,                           # full field, for the map texture
        },
        "danger_cells": danger,                     # only cells above the watch threshold
        "thresholds_mhz": {
            "watch": cfg["hazard"]["haf_watch_mhz"],
            "warning": cfg["hazard"]["haf_warning_mhz"],
            "severe": cfg["hazard"]["haf_severe_mhz"],
        },
        "notes": notes,
        "disclaimer": "Dayside HF-radio-blackout footprint of the current flare "
                      "(NOAA D-RAP). Sunlit hemisphere only; not the high-latitude "
                      "geomagnetic hazard.",
    }
