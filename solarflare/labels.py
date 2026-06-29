"""Flare-class taxonomy — the single source of truth for "what is a flare".

A solar flare's class is *defined* by its GOES 1-8 Angstrom peak X-ray flux:

    A  : flux < 1e-7  W/m^2
    B  : 1e-7 <= flux < 1e-6
    C  : 1e-6 <= flux < 1e-5
    M  : 1e-5 <= flux < 1e-4
    X  : flux >= 1e-4

Per the project brief, A/B and *weak* C flares are folded into "no-flare"
(they are operationally negligible). Upper-C, M and X are tracked
independently, and X-class events get an escalating severity warning.

Because the class is *defined* by flux, classifying a flare that is happening
*right now* needs no machine learning at all — it is a lookup. The ML lives in
*forecasting* future flares, not in identifying current ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import load_config

# Letter-class -> lower flux bound (W/m^2), used to parse NOAA codes like "M1.5".
_LETTER_BASE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def letter_to_flux(code: str | None) -> float:
    """Convert a NOAA flare code such as 'X2.3' or 'M1.0' to peak flux (W/m^2).

    Returns 0.0 for empty/unknown codes (treated as no event).
    """
    if not code:
        return 0.0
    code = code.strip().upper()
    letter = code[0]
    if letter not in _LETTER_BASE:
        return 0.0
    try:
        magnitude = float(code[1:]) if len(code) > 1 else 1.0
    except ValueError:
        magnitude = 1.0
    return _LETTER_BASE[letter] * magnitude


def flux_to_letter(flux: float) -> str:
    """Convert peak flux (W/m^2) to a NOAA letter code such as 'M1.5'."""
    if flux is None or flux <= 0 or math.isnan(flux):
        return "A0.0"
    for letter, base in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)):
        if flux >= base:
            return f"{letter}{flux / base:.1f}"
    return "A0.0"


@dataclass(frozen=True)
class Severity:
    level: str          # NONE | WATCH | WARNING | SEVERE | EXTREME
    label: str          # human sentence
    code: str           # NOAA-style R radio-blackout scale where relevant


def flux_to_category(flux: float, cfg: dict | None = None) -> str:
    """Map a peak flux to one of the operational categories: no-flare/C/M/X."""
    cfg = cfg or load_config()
    c = cfg["classes"]
    if flux is None or flux < c["noflare_below"]:
        return "no-flare"
    if flux < c["m_min"]:
        return "C"
    if flux < c["x_min"]:
        return "M"
    return "X"


def x_warning(flux: float, cfg: dict | None = None) -> Severity:
    """Escalating warning for X-class activity. Drives the big red banner."""
    cfg = cfg or load_config()
    c = cfg["classes"]
    if flux is None or flux < c["x_warn"]:
        return Severity("NONE", "No X-class activity.", "")
    if flux >= c["x_extreme"]:
        return Severity(
            "EXTREME",
            f"EXTREME X-class flare ({flux_to_letter(flux)}). "
            "Expect R3+ radio blackouts; HF comms and GNSS may be degraded.",
            "R3+",
        )
    if flux >= c["x_severe"]:
        return Severity(
            "SEVERE",
            f"SEVERE X-class flare ({flux_to_letter(flux)}). "
            "Strong radio blackout (R3) likely on the sunlit side.",
            "R3",
        )
    return Severity(
        "WARNING",
        f"X-class flare in progress ({flux_to_letter(flux)}). "
        "Radio blackout (R1-R2) possible.",
        "R1-R2",
    )


def probability_band(p: float, cfg: dict | None = None) -> dict:
    """Map a probability (e.g. P(M+)) to a named UI band: Low/Moderate/High/Severe.

    Sidesteps the hard yes/no threshold for display — the highest band whose
    `min` is <= p wins. Tone keys map to the severity colour scale in the UI.
    """
    cfg = cfg or load_config()
    bands = cfg.get("forecast_bands", [])
    if p is None or not bands:
        return {"name": "—", "tone": "quiet"}
    chosen = bands[0]
    for b in bands:
        if p >= b["min"]:
            chosen = b
    return {"name": chosen["name"], "tone": chosen.get("tone", "quiet")}


def binary_label_name(y: int) -> str:
    """Human name for the default binary task target."""
    return "M-or-greater flare expected" if int(y) == 1 else "no significant flare"
