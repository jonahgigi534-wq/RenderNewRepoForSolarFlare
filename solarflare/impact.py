"""Plain-language space-weather impact statements (STEP 8).

Maps the CURRENT NOAA R/S/G scale levels to rules-based plain-English impacts:
  R  radio blackouts        (solar X-rays; HF radio & navigation, sunlit side)
  S  solar radiation storms (energetic protons; poles, satellites, aviation)
  G  geomagnetic storms     (Earth's field; power grids, GPS, aurora; high lat)

A pure lookup over NOAA's own published scale values plus config-driven text — not
a new prediction. Output mirrors the other endpoints (status/notes). Never raises.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import sources
from .config import load_config

_SCALE_META = {
    "R": ("Radio blackouts", "HF radio & navigation — solar X-rays, sunlit side"),
    "S": ("Solar radiation storms", "Energetic protons — poles, satellites, aviation"),
    "G": ("Geomagnetic storms", "Earth's field — power grids, GPS, aurora; high latitudes"),
}


def _level(scales: dict, key: str) -> int:
    try:
        return int(scales.get("0", {}).get(key, {}).get("Scale") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def _historical(cfg: dict) -> list:
    """Config-driven real-event anchors (Quebec 1989, Starlink 2022, ...) — the
    'why it matters' context with documented costs. Whitespace in folded YAML
    strings is normalised."""
    out = []
    for e in cfg.get("impact", {}).get("historical_events", []) or []:
        try:
            out.append({"scale": str(e.get("scale", "")), "code": str(e.get("code", "")),
                        "year": int(e.get("year", 0)), "event": str(e.get("event", "")),
                        "detail": " ".join(str(e.get("detail", "")).split())})
        except (TypeError, ValueError):
            continue
    return out


def build_impact(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    res = sources.get_noaa_scales(cfg)
    scales = res.data if (res.ok and isinstance(res.data, dict)) else {}
    text = cfg["impact"]["statements"]
    out = {}
    for key, (name, domain) in _SCALE_META.items():
        lvl = _level(scales, key)
        stmts = text.get(key, {})
        plain = stmts.get(lvl, stmts.get(str(lvl), "—"))
        out[key] = {"scale": key, "level": lvl, "code": f"{key}{lvl}",
                    "name": name, "domain": domain, "active": lvl > 0,
                    "statement": plain}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": res.status if res.ok else "unavailable",
        "scales": out,
        "all_quiet": not any(v["active"] for v in out.values()),
        "historical": _historical(cfg),
        "disclaimer": "Plain-language impacts mapped from NOAA's R/S/G space-weather "
                      "scales. Guidance only — see NOAA SWPC for official alerts.",
    }
