"""Threshold alerting (STEP 8) — log + optional webhook. Email is OFF by default.

NO hardcoded secrets: the webhook URL comes from `alerts.webhook_url` in config OR
the `ALERT_WEBHOOK_URL` environment variable (env wins). If neither is set, only the
log channel fires. Email is intentionally not implemented (so no credentials live in
the repo). Safe to call on every forecast refresh — never raises to the caller.

Conditions (config-driven thresholds): NOAA R/S/G scale levels and the modelled
24 h storm probability. A real deployment would also de-duplicate repeats; here each
call reports the currently-active alerts.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from . import sources
from .config import load_config

log = logging.getLogger("helios.alerts")


def _webhook_url(cfg: dict) -> str:
    a = cfg.get("alerts", {})
    return os.environ.get("ALERT_WEBHOOK_URL") or a.get("webhook_url") or ""


def _scale_level(scales: dict, key: str) -> int:
    try:
        return int(scales.get("0", {}).get(key, {}).get("Scale") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def build_alerts(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    a = cfg.get("alerts", {})
    alerts = []

    scales = sources.get_noaa_scales(cfg)
    sd = scales.data if (scales.ok and isinstance(scales.data, dict)) else {}
    r, s, g = (_scale_level(sd, k) for k in ("R", "S", "G"))
    if r >= a.get("r_min", 3):
        alerts.append({"type": "radio_blackout", "level": f"R{r}",
                       "severity": "severe" if r >= 5 else "warning",
                       "message": f"R{r} radio blackout in progress — HF radio and "
                                  f"navigation degraded on the sunlit side."})
    if s >= a.get("s_min", 3):
        alerts.append({"type": "radiation_storm", "level": f"S{s}",
                       "severity": "severe" if s >= 5 else "warning",
                       "message": f"S{s} solar radiation storm — polar HF, satellites "
                                  f"and high-latitude aviation affected."})
    if g >= a.get("g_min", 4):
        alerts.append({"type": "geomagnetic_storm", "level": f"G{g}",
                       "severity": "severe" if g >= 5 else "warning",
                       "message": f"G{g} geomagnetic storm — power grids, GPS and "
                                  f"satellites at risk; aurora to lower latitudes."})

    # Modelled storm-forecast probability (resilient; storm path may be unavailable).
    try:
        from . import storm
        sf = storm.storm_forecast(cfg)
        p = sf.get("headline_p_storm")
        if p is not None and p >= a.get("storm_p_min", 0.5):
            alerts.append({"type": "storm_forecast", "level": f"{round(p * 100)}%",
                           "severity": "watch",
                           "message": f"Elevated storm forecast — {round(p * 100)}% "
                                      f"modelled chance of a G1+ storm within 24 h."})
    except Exception as exc:                              # noqa: BLE001
        log.debug("storm-forecast alert skipped: %s", exc)

    channels = _emit(cfg, alerts) if alerts else {"log": False, "webhook": False,
                                                  "email_enabled": bool(a.get("email_enabled", False))}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active": len(alerts),
        "alerts": alerts,
        "channels": channels,
        "all_clear": not alerts,
    }


def _emit(cfg: dict, alerts: list) -> dict:
    a = cfg.get("alerts", {})
    for al in alerts:                                     # log channel — always on
        log.warning("SPACE-WEATHER ALERT [%s] %s", al["level"], al["message"])
    out = {"log": True, "webhook": False, "email_enabled": bool(a.get("email_enabled", False))}
    url = _webhook_url(cfg)
    if a.get("webhook_enabled", True) and url:
        try:
            import requests
            requests.post(url, json={"source": "helios", "alerts": alerts},
                          timeout=cfg["live"]["request_timeout_s"])
            out["webhook"] = True
        except Exception as exc:                          # noqa: BLE001 (never raise)
            log.warning("alert webhook failed: %s", type(exc).__name__)
    return out
