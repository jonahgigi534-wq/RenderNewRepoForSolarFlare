"""Satellites-at-risk layer.

A flare-RISK INDICATOR, not a validated per-satellite prediction. We fetch
orbital data (TLE) from CelesTrak, compute each satellite's altitude band, and
assign a risk tier that scales with the CURRENT flare (NOAA R-scale + live
class) and the PREDICTED flare (existing forecast). The browser propagates the
TLEs with SGP4 (satellite.js) to place them at their real positions; the extra
"sunlit side" exposure bump is applied there, where positions are known.
"""
from __future__ import annotations

import math

from . import nowcast, sources
from .config import load_config

MU = 398600.4418      # km^3/s^2
RE = 6378.137         # km (equatorial)
RISK_TIERS = ["low", "elevated", "high", "severe"]


# ----------------------------------------------------------------------
# TLE parsing + altitude
# ----------------------------------------------------------------------
def _mean_altitude_km(line2: str) -> float | None:
    """Semi-major-axis altitude from the TLE mean motion (fixed columns)."""
    try:
        n_revday = float(line2[52:63])
        if n_revday <= 0:
            return None
        n = n_revday * 2 * math.pi / 86400.0
        a = (MU / (n * n)) ** (1 / 3)
        return a - RE
    except (ValueError, ZeroDivisionError):
        return None


def parse_tle(text: str) -> list[dict]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    sats, i = [], 0
    while i + 2 < len(lines) + 1 and i + 2 <= len(lines):
        if i + 2 >= len(lines):
            break
        name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            i += 1
            continue
        alt = _mean_altitude_km(l2)
        if alt is None:
            i += 3
            continue
        try:
            norad = int(l2[2:7])
        except ValueError:
            norad = None
        sats.append({"name": name, "norad": norad, "line1": l1, "line2": l2,
                     "alt_km": round(alt, 1)})
        i += 3
    return sats


def band_for(alt_km: float, cfg: dict) -> str:
    for b in cfg["satellites"]["bands"]:
        if b["min"] <= alt_km < b["max"]:
            return b["name"]
    return "beyond"


# ----------------------------------------------------------------------
# Flare context + risk
# ----------------------------------------------------------------------
def _flare_context(cfg: dict) -> dict:
    """Compact current+predicted flare drivers (cheap; a few cached fetches)."""
    scales = sources.get_noaa_scales(cfg)
    r_scale = 0
    if scales.ok and isinstance(scales.data, dict):
        try:
            r_scale = int(scales.data.get("0", {}).get("R", {}).get("Scale") or 0)
        except (ValueError, TypeError):
            r_scale = 0
    flux = sources.get_xray_flux(cfg)
    now = nowcast.nowcast(flux.data, cfg) if flux.ok else {"available": False}
    pred = sources.noaa_region_forecast(cfg)
    pred_pM = pred.get("p_M_or_greater_24h") if pred.get("available") else None
    return {
        "r_scale": r_scale,
        "current_class": now.get("current_class") if now.get("available") else None,
        "predicted_pM_24h": pred_pM,
        "subsolar_driven": True,
    }


def risk_tier(band: str, ctx: dict) -> tuple[str, str]:
    """Return (tier, reason). Heuristic, config-light, honest."""
    r = ctx["r_scale"]
    pM = ctx["predicted_pM_24h"]
    score = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}.get(r, 0)
    reason_bits = []
    if r > 0:
        reason_bits.append(f"R{r} radio blackout in progress")
    if pM is not None and pM >= 0.75:
        score += 1
        reason_bits.append(f"{round(pM*100)}% chance of M+ flare in 24h")
    elif pM is not None and pM >= 0.5:
        score = max(score, 1)
        reason_bits.append(f"{round(pM*100)}% chance of M+ flare in 24h")
    # Higher orbits sit outside the inner magnetosphere -> more SEP exposure.
    if band in ("MEO", "GEO", "beyond") and r >= 3:
        score += 1
        reason_bits.append(f"{band} altitude exposed to enhanced energetic particles")
    elif band == "LEO" and r >= 2:
        reason_bits.append("LEO drag and HF/GNSS effects elevated")
    tier = RISK_TIERS[min(score, 3)]
    if not reason_bits:
        reason_bits.append("Quiet Sun; nominal radiation environment")
    return tier, "; ".join(reason_bits)


# ----------------------------------------------------------------------
# Public builder
# ----------------------------------------------------------------------
def _collect(scope: str, cfg: dict) -> tuple[list[dict], list[str], str]:
    s = cfg["satellites"]
    notes, status = [], "live"
    if scope == "all":
        res = sources.get_celestrak_tle(s["all_group"], cfg)
        notes.extend(res.notes)
        sats = parse_tle(res.data) if res.ok else []
        status = res.status if res.ok else "unavailable"
        return sats[: s["max_all"]], notes, status
    # default: a curated subset
    seen, sats = set(), []
    for grp in s["default_groups"]:
        res = sources.get_celestrak_tle(grp, cfg)
        if not res.ok:
            notes.append(f"{grp} unavailable")
            status = "partial"
            continue
        for sat in parse_tle(res.data):
            if sat["norad"] in seen:
                continue
            seen.add(sat["norad"]); sats.append(sat)
    star = sources.get_celestrak_tle(s["starlink_group"], cfg)
    if star.ok:
        for sat in parse_tle(star.data)[: s["starlink_sample"]]:
            if sat["norad"] not in seen:
                seen.add(sat["norad"]); sats.append(sat)
    return sats[: s["max_default"]], notes, status


def build_satellites(scope: str = "default", cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    from datetime import datetime, timezone
    sats, notes, status = _collect(scope, cfg)
    ctx = _flare_context(cfg)
    counts = {t: 0 for t in RISK_TIERS}
    for sat in sats:
        sat["band"] = band_for(sat["alt_km"], cfg)
        tier, reason = risk_tier(sat["band"], ctx)
        sat["risk"] = tier
        sat["reason"] = reason
        counts[tier] += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "status": status,
        "count": len(sats),
        "flare_context": ctx,
        "risk_counts": counts,
        "satellites": sats,
        "notes": notes,
        "disclaimer": "Flare-risk INDICATOR by altitude band and current/predicted "
                      "flare level — not a validated per-satellite prediction. "
                      "Positions propagated client-side via SGP4.",
    }
