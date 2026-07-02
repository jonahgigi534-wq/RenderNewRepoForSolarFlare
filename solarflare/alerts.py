"""Threshold alerting (STEP 8) — log + optional webhook. Email is OFF by default.

NO hardcoded secrets: the webhook URL comes from `alerts.webhook_url` in config OR
the `ALERT_WEBHOOK_URL` environment variable (env wins). If neither is set, only the
log channel fires. `demo_alert` also exercises email via solarflare.notify — dry-run
(logged to the outbox, not sent) unless $SMTP_PASSWORD is set, and real sends
additionally require $DEMO_ALERT_TOKEN so a public deploy's demo button can't be
used to spam the recipients. Safe to call on every forecast refresh — never raises
to the caller.

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


def demo_alert(cfg: dict | None = None) -> dict:
    """Fire ONE clearly-labelled demo alert through the REAL channels (log +
    webhook + email via the notifier). Proves the alert path works end to end
    without waiting for the Sun to cooperate. Dry-run safe: with SMTP unset the
    email lands in the outbox log instead. Never raises."""
    cfg = cfg or load_config()
    al = {"type": "demo", "level": "DEMO", "severity": "demo",
          "message": "DEMO ALERT — this is what a live Helios alert looks like: "
                     "the flare model's P(M-class+ within 24 h) crossed the alert "
                     "threshold. HF radio degradation likely on the sunlit side. "
                     "(Demonstration only — no real event.)"}
    channels = _emit(cfg, [al])
    try:
        from . import notify
        if os.environ.get("SMTP_PASSWORD") and not os.environ.get("DEMO_ALERT_TOKEN"):
            # Public-abuse guard: live SMTP + an unauthenticated endpoint would
            # let anyone on the internet email the recipients. Real sends
            # require the operator to configure DEMO_ALERT_TOKEN (the endpoint
            # then enforces the matching header).
            channels["email"] = "skipped (live SMTP requires DEMO_ALERT_TOKEN)"
        else:
            channels["email"] = notify.send_email(
                cfg, "[HELIOS DEMO] Space-weather alert — demonstration",
                al["message"] + "\n\nSent by the Helios demo-alert button to show the "
                "real alert pipeline (log + webhook + email) working end to end.")
    except Exception as exc:                              # noqa: BLE001 (never raise)
        log.warning("demo alert email failed: %s", type(exc).__name__)
        channels["email"] = "failed"
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "demo": True, "alert": al, "channels": channels}


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
