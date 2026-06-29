"""Statistical near-term flare forecaster from recent GOES X-ray flux.

This is the always-on forecast track. It does NOT need the SWAN-SF model and
works the moment NOAA returns data. It is deliberately a transparent,
well-understood baseline rather than a black box:

  * Persistence + Poisson recurrence: recent flaring strongly predicts
    near-term flaring (flares cluster by active region). We estimate the
    hourly rate of M+ events from the last 24-48h and convert it to the
    probability of >=1 event in the next 12/24/48h via 1 - exp(-rate*hours).
  * A "current elevation" boost: if the background flux is already elevated
    (rising trend, recent C/M activity) the rate is scaled up.

Honest framing: this is a climatological/persistence skill baseline. It is the
standard yardstick every operational flare model is measured against, and on
its own it is genuinely useful for 12-24h lead time. The SWAN-SF SHARP model
(when magnetic features are available) is ensembled on top for extra skill.
"""
from __future__ import annotations

import math

from .config import load_config
from . import labels


def _prob_at_least_one(rate_per_hour: float, hours: float) -> float:
    """Poisson P(N>=1) over `hours` given an hourly event rate."""
    return 1.0 - math.exp(-max(rate_per_hour, 0.0) * hours)


def forecast(activity: dict, cfg: dict | None = None) -> dict:
    """Return per-horizon probabilities for M+ and X-class events.

    `activity` comes from nowcast.activity_features().
    """
    cfg = cfg or load_config()
    if not activity:
        return {"available": False, "reason": "no recent flux"}

    # Flux-class boundaries come from the central taxonomy so retuning
    # config.classes propagates here instead of drifting from labels.py.
    classes = cfg.get("classes", {})
    m_min = classes.get("m_min", 1e-5)
    upperc_min = classes.get("upperC_min", 5e-6)
    x_min = classes.get("x_min", 1e-4)

    # Base hourly M+ rate from the last 24h (events / 24h). Smooth with a small
    # prior so a totally quiet sun still yields a tiny, non-zero probability.
    n_m_24 = activity.get("n_M_24h", 0)
    n_c_24 = activity.get("n_C_24h", 0)
    base_m_rate = (n_m_24 + 0.10) / 24.0           # +0.10 climatological prior
    # C-activity is a leading indicator of M flares; fold it in gently.
    base_m_rate += (n_c_24 / 24.0) * 0.05

    # Elevation / trend boost.
    trend = activity.get("trend_ratio", 1.0)
    max24 = activity.get("max_24h", 0.0)
    boost = 1.0
    if trend > 1.5:
        boost *= 1.4
    if max24 >= m_min:       # an M flare already happened today
        boost *= 1.6
    elif max24 >= upperc_min:  # upper-C activity
        boost *= 1.2
    m_rate = base_m_rate * boost

    # X-rate is empirically ~1/10 of the M-rate, nudged by big recent events.
    x_rate = m_rate * 0.10
    if max24 >= x_min:
        x_rate *= 2.0

    horizons = {}
    for h in (12, 24, 48):
        p_m = _prob_at_least_one(m_rate, h)
        p_x = _prob_at_least_one(x_rate, h)
        # P(C+) is higher than P(M+); approximate from C-rate.
        c_rate = max((n_c_24 + 0.5) / 24.0, m_rate)
        p_c = _prob_at_least_one(c_rate, h)
        horizons[f"{h}h"] = {
            "p_C_or_greater": round(min(p_c, 0.99), 4),
            "p_M_or_greater": round(min(p_m, 0.99), 4),
            "p_X_class": round(min(p_x, 0.99), 4),
            "expected_max_class": _expected_class(p_m, p_x),
        }
    return {"available": True, "model": "flux-persistence-poisson", "horizons": horizons}


def _expected_class(p_m: float, p_x: float) -> str:
    if p_x >= 0.5:
        return "X"
    if p_m >= 0.5:
        return "M"
    if p_m >= 0.15:
        return "C-M"
    return "no-flare"
