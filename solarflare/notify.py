"""Live email notifier — predictive flare alerts + post-event verification.

Three jobs, run each cycle by `run_once()`:

  1. PREDICTIVE ALERT — when the live forecast crosses the configured magnitude +
     probability threshold (e.g. >= 50% chance of an M-class flare in 24 h) and we
     have not already warned for this episode, email a warning with the forecast
     details and record the prediction (with a verify-after time).
  2. POST-EVENT VERIFICATION — for predictions whose observation window has elapsed,
     query the authoritative NOAA SWPC GOES X-ray *flare-event* log to find the
     actual peak flare class inside the window.
  3. ACCURACY FOLLOW-UP — email a follow-up stating whether the flare materialised
     or was a false positive, including the recorded peak magnitude; mark resolved.

State lives in a local SQLite DB (`<cache_dir>/notify.db`). Email goes out via SMTP
using credentials from the ENVIRONMENT (`SMTP_PASSWORD`) plus config — **no secrets
are committed**. With SMTP unset it runs in DRY-RUN mode: every message is appended
to `<cache_dir>/notify_outbox.log` instead of being sent, so the whole pipeline is
testable without credentials. Nothing here raises to the caller.

Run continuously:   python -m solarflare.notify
One cycle:          python -m solarflare.notify --once
Show state:         python -m solarflare.notify --status
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from . import labels, predictor, sources
from .config import load_config

log = logging.getLogger("helios.notify")
_bg_started = False


# ----------------------------------------------------------------------
# Paths + DB
# ----------------------------------------------------------------------
def _db_path(cfg: dict) -> str:
    n = cfg.get("notify", {})
    return n.get("db_path") or os.path.join(cfg["paths"]["cache_dir"], "notify.db")


def _outbox_path(cfg: dict) -> str:
    n = cfg.get("notify", {})
    return n.get("outbox_log") or os.path.join(cfg["paths"]["cache_dir"], "notify_outbox.log")


def _db(cfg: dict) -> sqlite3.Connection:
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
    conn = sqlite3.connect(_db_path(cfg), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, threshold TEXT, probability REAL, expected_class TEXT,
            window_h INTEGER, verify_after TEXT,
            status TEXT,                 -- pending | verified
            alert_status TEXT,           -- sent | dry-run | failed
            materialized INTEGER,        -- 0/1, NULL until verified
            actual_peak_class TEXT, actual_peak_flux REAL,
            verified_at TEXT, followup_status TEXT
        )""")
    conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    for col, decl in (("kind", "TEXT"), ("p_m", "REAL"), ("p_x", "REAL"),
                      ("current_class", "TEXT"),
                      # peak-magnitude forecast + honest log-flux grading (additive)
                      ("p_c", "REAL"), ("pred_peak_flux", "REAL"),
                      ("pred_peak_class", "TEXT"), ("persist_pred_flux", "REAL"),
                      ("clim_pred_flux", "REAL"), ("err_dex", "REAL"),
                      ("persist_err_dex", "REAL"), ("clim_err_dex", "REAL")):
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


def _meta_get(conn: sqlite3.Connection, key: str):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _to_ct(ts) -> str:
    """Render a UTC timestamp in US Central time, DST-aware (shows CST or CDT).
    Underlying data stays UTC; this is display-only for the spreadsheet."""
    dt = _parse(ts)
    if dt is None:
        return str(ts) if ts else ""
    try:
        from zoneinfo import ZoneInfo
        return dt.astimezone(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:                                     # noqa: BLE001 (fallback: raw UTC)
        return str(ts)


# ----------------------------------------------------------------------
# Email (dry-run safe; no secrets in code)
# ----------------------------------------------------------------------
def _outbox(cfg: dict, to: str, subject: str, body: str, disposition: str) -> None:
    try:
        with open(_outbox_path(cfg), "a", encoding="utf-8") as fh:
            fh.write(f"\n{'='*70}\n[{_now().isoformat()}] {disposition} -> {to or '(no recipient)'}\n"
                     f"Subject: {subject}\n{'-'*70}\n{body}\n")
    except OSError:
        pass


def _recipients(cfg: dict) -> list:
    """The list of recipient addresses (notify.recipients list, or a comma-separated
    notify.recipient string for back-compat)."""
    n = cfg.get("notify", {})
    # Env override (HELIOS_RECIPIENTS, comma-separated) wins, so addresses can be
    # kept out of the committed config; falls back to notify.recipients.
    rec = os.environ.get("HELIOS_RECIPIENTS") or n.get("recipients") or n.get("recipient") or []
    if isinstance(rec, str):
        rec = [x.strip() for x in rec.split(",")]
    return [r for r in rec if r]


def send_email(cfg: dict, subject: str, body: str, to=None) -> str:
    """Send via SMTP to ALL configured recipients, or DRY-RUN to the outbox log if
    not fully configured. Every message is recorded in the outbox for an audit
    trail. Never raises. Returns 'sent' | 'dry-run' | 'failed'."""
    n = cfg.get("notify", {})
    rec = ([to] if isinstance(to, str) else to) if to is not None else _recipients(cfg)
    to_hdr = ", ".join(rec)
    host = n.get("smtp_host", "")
    # Env overrides (HELIOS_SMTP_USER / HELIOS_SENDER) so the sending identity can
    # also stay out of committed config; both fall back to the config values.
    user = os.environ.get("HELIOS_SMTP_USER") or n.get("smtp_user", "")
    sender = os.environ.get("HELIOS_SENDER") or n.get("sender") or user or "helios-notifier@localhost"
    pw = os.environ.get("SMTP_PASSWORD", "")

    if not (host and rec and pw):                            # not configured -> dry-run
        _outbox(cfg, to_hdr, subject, body,
                "DRY-RUN (set notify.smtp_host, notify.smtp_user, recipients and $SMTP_PASSWORD to send)")
        return "dry-run"
    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = sender, to_hdr, subject
        msg.set_content(body)
        port = int(n.get("smtp_port", 587))
        if n.get("smtp_use_tls", True):
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.starttls(); s.login(user, pw); s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, port, timeout=20) as s:
                s.login(user, pw); s.send_message(msg)
        _outbox(cfg, to_hdr, subject, body, "SENT")
        log.info("notifier email sent to %s: %s", to_hdr, subject)
        return "sent"
    except Exception as exc:                                  # noqa: BLE001 (never raise)
        log.warning("notifier email failed: %s", type(exc).__name__)
        _outbox(cfg, to_hdr, subject, body, f"FAILED ({type(exc).__name__}); not sent")
        return "failed"


# ----------------------------------------------------------------------
# Forecast state + verification
# ----------------------------------------------------------------------
def _forecast_state(cfg: dict) -> dict:
    """Pull the current flare forecast into the numbers the trigger needs."""
    r = predictor.predict(cfg=cfg)
    fc = r.get("forecast", {})
    ens = fc.get("ensemble_24h", {}) or {}
    h24 = (fc.get("flux_track", {}).get("horizons", {}) or {}).get("24h", {}) or {}
    nc = r.get("nowcast", {}) or {}
    return {
        "generated_at": r.get("generated_at"),
        "data_status": (r.get("data_freshness", {}) or {}).get("status"),
        "p_M": ens.get("p_M_or_greater_24h", h24.get("p_M_or_greater")),
        "p_X": h24.get("p_X_class"),
        "p_C": h24.get("p_C_or_greater"),               # for the peak-magnitude forecast
        "expected_class": h24.get("expected_max_class"),
        "current_class": nc.get("current_class"),
        "peak_24h_class": nc.get("peak_24h_class"),
        "peak_24h_flux": nc.get("peak_24h_flux_wm2"),   # persistence anchor (numeric)
    }


def _threshold_flux(cfg: dict, threshold: str) -> float:
    c = cfg["classes"]
    return c["x_min"] if threshold.upper() == "X" else c["m_min"]


def actual_peak_in_window(cfg: dict, start: datetime, end: datetime) -> dict:
    """Authoritative peak flare in [start, end] from NOAA's GOES flare-event log
    (falls back to peak GOES X-ray flux if the event list is unavailable)."""
    res = sources.get_recent_flares(cfg)
    if res.ok and isinstance(res.data, list):
        peak_flux, peak_class, peak_time = 0.0, None, None
        for ev in res.data:
            t = _parse(ev.get("max_time") or ev.get("begin_time"))
            if t is None or not (start <= t <= end):
                continue
            flux = ev.get("max_xrlong")
            try:
                flux = float(flux) if flux is not None else labels.letter_to_flux(ev.get("max_class"))
            except (TypeError, ValueError):
                flux = labels.letter_to_flux(ev.get("max_class"))
            if flux > peak_flux:
                peak_flux, peak_class, peak_time = flux, ev.get("max_class"), ev.get("max_time")
        return {"source": "noaa-goes-flare-events", "status": res.status,
                "peak_flux": peak_flux,
                "peak_class": peak_class or labels.flux_to_letter(peak_flux),
                "peak_time": peak_time, "n_events": len(res.data)}

    flux_res = sources.get_xray_flux(cfg)                     # fallback: raw flux series
    if flux_res.ok and isinstance(flux_res.data, list):
        peak_flux, peak_time = 0.0, None
        for row in flux_res.data:
            t = _parse(row.get("time_tag"))
            f = row.get("flux")
            if t is None or f is None or not (start <= t <= end):
                continue
            if float(f) > peak_flux:
                peak_flux, peak_time = float(f), row.get("time_tag")
        return {"source": "noaa-goes-xray-flux", "status": flux_res.status,
                "peak_flux": peak_flux, "peak_class": labels.flux_to_letter(peak_flux),
                "peak_time": peak_time}
    return {"source": "unavailable", "status": "unavailable",
            "peak_flux": 0.0, "peak_class": "A0.0", "peak_time": None}


# ----------------------------------------------------------------------
# Email bodies
# ----------------------------------------------------------------------
def _alert_email(cfg: dict, st: dict, threshold: str, prob: float, window_h: int) -> tuple[str, str]:
    pct = round(prob * 100)
    subject = f"[Helios] Flare WARNING — {pct}% chance of an {threshold}-class+ flare in {window_h} h"
    body = (
        f"HELIOS SOLAR-FLARE FORECAST ALERT\n"
        f"Issued: {st.get('generated_at')}  (data: {st.get('data_status')})\n\n"
        f"The model forecasts a {pct}% chance of an {threshold.upper()}-class or "
        f"greater flare within the next {window_h} hours.\n\n"
        f"  Forecast P(M+ in 24h) : {('%.0f%%' % (st['p_M']*100)) if st.get('p_M') is not None else '—'}\n"
        f"  Forecast P(X in 24h)  : {('%.0f%%' % (st['p_X']*100)) if st.get('p_X') is not None else '—'}\n"
        f"  Expected peak class   : {st.get('expected_class') or '—'}\n"
        f"  Current conditions    : {st.get('current_class') or '—'} "
        f"(24h peak {st.get('peak_24h_class') or '—'})\n\n"
        f"A verification follow-up will be sent after the {window_h}-hour observation "
        f"window, confirming whether the flare materialised (with its recorded peak "
        f"magnitude from NOAA GOES) or was a false alarm.\n\n"
        f"— Helios. Probabilistic guidance; see https://www.swpc.noaa.gov/ for official alerts."
    )
    return subject, body


def _followup_email(cfg: dict, row, peak: dict, materialized: bool) -> tuple[str, str]:
    thr = row["threshold"]
    verdict = "CONFIRMED" if materialized else "FALSE POSITIVE"
    subject = f"[Helios] Verification: {thr}-class forecast {verdict} — peak {peak['peak_class']}"
    pct = round((row["probability"] or 0) * 100)
    body = (
        f"HELIOS FORECAST VERIFICATION\n"
        f"Prediction issued : {row['created_at']}\n"
        f"Observation window: {row['window_h']} h (through {row['verify_after']})\n\n"
        f"Predicted: {pct}% chance of an {thr}-class+ flare.\n"
        f"Outcome  : {verdict}.\n\n"
        f"  Recorded peak flare : {peak['peak_class']}"
        f"{(' at ' + peak['peak_time']) if peak.get('peak_time') else ''}\n"
        f"  Peak X-ray flux     : {peak['peak_flux']:.2e} W/m^2\n"
        f"  Source              : {peak['source']} ({peak.get('status')})\n\n"
        + ("The forecast flare DID occur within the window.\n"
           if materialized else
           "No flare of the predicted magnitude occurred within the window "
           "(false positive).\n")
        + "\n— Helios automatic verification against NOAA SWPC GOES flare data."
    )
    return subject, body


# ----------------------------------------------------------------------
# Peak-magnitude forecast (attached to each logged prediction; graded on verify)
# ----------------------------------------------------------------------
def _record_magnitude(cfg: dict, conn: sqlite3.Connection, row_id: int, st: dict) -> None:
    """Attach the next-24h peak-magnitude forecast + its two baselines to a row.
    Additive and best-effort: a failure here never disrupts the notify cycle."""
    try:
        from . import magnitude
        mg = magnitude.predict(st, cfg)
        conn.execute(
            "UPDATE predictions SET p_c=?, pred_peak_flux=?, pred_peak_class=?, "
            "persist_pred_flux=?, clim_pred_flux=? WHERE id=?",
            (st.get("p_C"), mg["model_flux"], mg["model_class"],
             mg["persistence_flux"], mg["climatology_flux"], row_id))
        conn.commit()
    except Exception as exc:                                  # noqa: BLE001 (never raise)
        log.debug("magnitude forecast record skipped: %s", exc)


def _grade_magnitude(cfg: dict, conn: sqlite3.Connection, row, actual_flux) -> None:
    """After the window, score the magnitude forecast + baselines in log-flux error
    (dex) against the actual peak. Best-effort; rows logged before this feature
    (no stored prediction) simply stay ungraded."""
    try:
        from . import magnitude
        keys = row.keys()
        pred = row["pred_peak_flux"] if "pred_peak_flux" in keys else None
        persist = row["persist_pred_flux"] if "persist_pred_flux" in keys else None
        clim = row["clim_pred_flux"] if "clim_pred_flux" in keys else None
        conn.execute(
            "UPDATE predictions SET err_dex=?, persist_err_dex=?, clim_err_dex=? WHERE id=?",
            (magnitude.error_dex(pred, actual_flux),
             magnitude.error_dex(persist, actual_flux),
             magnitude.error_dex(clim, actual_flux), row["id"]))
        conn.commit()
    except Exception as exc:                                  # noqa: BLE001 (never raise)
        log.debug("magnitude grade skipped: %s", exc)


# ----------------------------------------------------------------------
# The three jobs
# ----------------------------------------------------------------------
def check_and_alert(cfg: dict, conn: sqlite3.Connection) -> dict:
    n = cfg["notify"]
    threshold = str(n.get("alert_threshold", "M")).upper()
    need_p = float(n.get("alert_probability", 0.5))
    window_h = int(n.get("observation_window_h", 24))
    cooldown_h = float(n.get("alert_cooldown_h", 12))

    st = _forecast_state(cfg)
    prob = st["p_X"] if threshold == "X" else st["p_M"]
    if prob is None:
        return {"triggered": False, "reason": "no forecast probability"}
    if prob < need_p:
        return {"triggered": False, "p": round(prob, 3), "need": need_p, "threshold": threshold}

    # De-duplicate: skip if we already alerted for this threshold within the cooldown.
    cutoff = (_now() - timedelta(hours=cooldown_h)).isoformat()
    recent = conn.execute(
        "SELECT created_at FROM predictions WHERE threshold=? AND created_at>=? "
        "ORDER BY created_at DESC LIMIT 1", (threshold, cutoff)).fetchone()
    if recent:
        return {"triggered": False, "reason": "within cooldown", "p": round(prob, 3),
                "last_alert": recent["created_at"]}

    subject, body = _alert_email(cfg, st, threshold, prob, window_h)
    disposition = send_email(cfg, subject, body)
    verify_after = (_now() + timedelta(hours=window_h)).isoformat()
    cur = conn.execute(
        "INSERT INTO predictions(kind,created_at,threshold,probability,p_m,p_x,"
        "expected_class,current_class,window_h,verify_after,status,alert_status) "
        "VALUES ('alert',?,?,?,?,?,?,?,?,?, 'pending', ?)",
        (_now().isoformat(), threshold, prob, st.get("p_M"), st.get("p_X"),
         st.get("expected_class"), st.get("current_class"), window_h, verify_after, disposition))
    conn.commit()
    _record_magnitude(cfg, conn, cur.lastrowid, st)          # attach the peak-magnitude forecast
    log.warning("notifier ALERT (%s) id=%s p=%.2f -> %s", threshold, cur.lastrowid, prob, disposition)
    return {"triggered": True, "id": cur.lastrowid, "threshold": threshold,
            "p": round(prob, 3), "alert_status": disposition, "verify_after": verify_after}


def verify_pending(cfg: dict, conn: sqlite3.Connection) -> list:
    now_iso = _now().isoformat()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE status='pending' AND verify_after<=?",
        (now_iso,)).fetchall()
    out = []
    for row in rows:
        start = _parse(row["created_at"]) or (_now() - timedelta(hours=row["window_h"]))
        end = _parse(row["verify_after"]) or _now()
        peak = actual_peak_in_window(cfg, start, end)
        materialized = peak["peak_flux"] >= _threshold_flux(cfg, row["threshold"])
        _grade_magnitude(cfg, conn, row, peak["peak_flux"])  # honest log-flux skill grade
        kind = (row["kind"] if "kind" in row.keys() else None) or "alert"
        if kind == "alert":                                  # only alerts get a follow-up email
            subject, body = _followup_email(cfg, row, peak, materialized)
            disposition = send_email(cfg, subject, body)
        else:                                                # daily log: record only, no email
            disposition = "logged"
        conn.execute(
            "UPDATE predictions SET status='verified', materialized=?, actual_peak_class=?,"
            " actual_peak_flux=?, verified_at=?, followup_status=? WHERE id=?",
            (1 if materialized else 0, peak["peak_class"], peak["peak_flux"],
             now_iso, disposition, row["id"]))
        conn.commit()
        log.warning("notifier VERIFY id=%s kind=%s materialized=%s peak=%s -> %s",
                    row["id"], kind, materialized, peak["peak_class"], disposition)
        out.append({"id": row["id"], "kind": kind, "threshold": row["threshold"],
                    "materialized": materialized, "peak_class": peak["peak_class"],
                    "followup_status": disposition})
    return out


def _pct(p) -> str:
    return f"{round(p*100)}%" if isinstance(p, (int, float)) else "—"


def daily_digest(cfg: dict) -> tuple[str, str]:
    """A full daily space-weather forecast email (flares + storm + current scales)
    — not just a threshold alert. Each section is resilient on its own."""
    today = _now().strftime("%Y-%m-%d")
    L = [f"HELIOS — DAILY SPACE-WEATHER FORECAST   ({today} UTC)",
         "=" * 56, ""]

    try:                                                     # --- solar flares ---
        r = predictor.predict(cfg=cfg)
        nc = r.get("nowcast", {}) or {}
        h = ((r.get("forecast", {}) or {}).get("flux_track", {}) or {}).get("horizons", {}) or {}
        ens = (r.get("forecast", {}) or {}).get("ensemble_24h", {}) or {}
        L += ["SOLAR FLARES",
              f"  Now: {nc.get('current_class', '—')} "
              f"({'FLARING' if nc.get('is_flaring') else 'quiet'}); 24h peak {nc.get('peak_24h_class', '—')}",
              f"  Ensemble chance of an M-class+ flare (24h): {_pct(ens.get('p_M_or_greater_24h'))}",
              "  By lead time:"]
        for k in ("12h", "24h", "48h"):
            hz = h.get(k) or {}
            L.append(f"     {k:>4}:  C+ {_pct(hz.get('p_C_or_greater'))}   "
                     f"M+ {_pct(hz.get('p_M_or_greater'))}   X {_pct(hz.get('p_X_class'))}   "
                     f"(expected peak {hz.get('expected_max_class', '—')})")
        L.append(f"  Active regions on disk: {r.get('active_region_count', '—')}")
        w = (nc.get("x_warning") or {})
        if w.get("level") and w["level"] != "NONE":
            L.append(f"  ** {w['level']}: {w.get('message', '')}")
    except Exception:                                        # noqa: BLE001
        L.append("SOLAR FLARES\n  (forecast unavailable this run)")
    L.append("")

    try:                                                     # --- geomagnetic storm ---
        from . import storm
        sf = storm.storm_forecast(cfg)
        L += ["GEOMAGNETIC STORM",
              f"  Modelled chance of a G1+ storm (Kp>=5, 24h): "
              f"{_pct(sf.get('headline_p_storm'))} ({sf.get('headline_source')})",
              f"  NOAA predicted peak Kp: {sf.get('forecast_kp', '—')} "
              f"(G{sf.get('forecast_g_scale', 0)})"]
    except Exception:                                        # noqa: BLE001
        L.append("GEOMAGNETIC STORM\n  (forecast unavailable this run)")
    L.append("")

    try:                                                     # --- current impacts (R/S/G) ---
        from . import impact
        sc = impact.build_impact(cfg).get("scales", {})
        L.append("CURRENT NOAA SCALES")
        for k in ("R", "S", "G"):
            v = sc.get(k) or {}
            L.append(f"  {v.get('code', '—')}: {v.get('statement', '')}")
    except Exception:                                        # noqa: BLE001
        pass

    L += ["", "— Helios automatic daily forecast. Probabilistic guidance; see "
          "https://www.swpc.noaa.gov/ for official products."]
    return f"[Helios] Daily space-weather forecast — {today}", "\n".join(L)


def maybe_send_daily(cfg: dict, conn: sqlite3.Connection) -> dict:
    """Send the daily forecast once per day, at/after the configured UTC hour."""
    n = cfg["notify"]
    if not n.get("daily_forecast_enabled", True):
        return {"sent": False, "reason": "disabled"}
    now = _now()
    today = now.date().isoformat()
    if _meta_get(conn, "last_digest_date") == today:
        return {"sent": False, "reason": "already sent today"}
    if now.hour < int(n.get("daily_send_hour_utc", 13)):
        return {"sent": False, "reason": "before send hour"}
    if _meta_get(conn, "last_digest_attempt") == today:
        return {"sent": False, "reason": "attempted today (dry-run); will send once live"}
    subject, body = daily_digest(cfg)
    disposition = send_email(cfg, subject, body)
    # Only a real send marks the day done; a dry-run only records the attempt so the
    # live run can still send today.
    _meta_set(conn, "last_digest_date" if disposition == "sent" else "last_digest_attempt", today)
    log.warning("notifier DAILY forecast -> %s", disposition)
    return {"sent": disposition == "sent", "disposition": disposition, "date": today}


def log_daily_forecast(cfg: dict, conn: sqlite3.Connection) -> dict:
    """Record ONE forecast snapshot per day (kind='daily') so the spreadsheet has a
    continuous prediction history to compare against actuals — not only the alerts."""
    n = cfg["notify"]
    now = _now()
    today = now.date().isoformat()
    if _meta_get(conn, "last_forecast_log_date") == today:
        return {"logged": False, "reason": "already logged today"}
    st = _forecast_state(cfg)
    if st.get("p_M") is None:
        return {"logged": False, "reason": "no forecast"}
    thr = str(n.get("alert_threshold", "M")).upper()
    window_h = int(n.get("observation_window_h", 24))
    verify_after = (now + timedelta(hours=window_h)).isoformat()
    p_alert = st.get("p_X") if thr == "X" else st.get("p_M")
    cur = conn.execute(
        "INSERT INTO predictions(kind,created_at,threshold,probability,p_m,p_x,"
        "expected_class,current_class,window_h,verify_after,status,alert_status) "
        "VALUES('daily',?,?,?,?,?,?,?,?,?, 'pending','n/a')",
        (now.isoformat(), thr, p_alert, st.get("p_M"), st.get("p_X"),
         st.get("expected_class"), st.get("current_class"), window_h, verify_after))
    conn.commit()
    _record_magnitude(cfg, conn, cur.lastrowid, st)          # attach the peak-magnitude forecast
    _meta_set(conn, "last_forecast_log_date", today)
    return {"logged": True, "date": today}


# ----------------------------------------------------------------------
# Spreadsheet (CSV) export — full prediction history vs actual outcomes
# ----------------------------------------------------------------------
_CSV_COLS = ["id", "kind", "issued_ct", "forecast_threshold", "p_alert", "p_M_24h",
             "p_X_24h", "expected_class", "current_class_at_issue", "window_h",
             "verify_after_ct", "status", "actual_peak_class", "actual_peak_flux_wm2",
             "outcome", "outcome_detail", "verified_ct", "alert_email", "followup",
             # peak-magnitude forecast + honest log-flux grade
             "predicted_peak_class", "predicted_peak_flux_wm2", "err_dex",
             "persistence_err_dex", "climatology_err_dex",
             "beat_persistence", "beat_climatology"]


def _beat(err, base_err) -> str:
    """Did the model's log-flux error beat a baseline's? '' until both are graded."""
    if err is None or base_err is None:
        return ""
    return "yes" if err <= base_err else "no"


def _row_record(r) -> dict:
    d = dict(r)
    rd = lambda v: round(v, 4) if isinstance(v, (int, float)) else v
    rd3 = lambda v: round(v, 3) if isinstance(v, (int, float)) else v
    # `outcome` is the simple accuracy call: HIT if the forecast's >=50% side
    # matched what happened, MISS otherwise — same two labels for every kind.
    # `outcome_detail` keeps the full 2x2 contingency-table term (HIT/MISS/
    # FALSE_ALARM/CORRECT_REJECTION) so the "why" isn't lost: a MISS in
    # `outcome` can be either an unwarned event (contingency MISS) or a false
    # alarm (contingency FALSE_ALARM), and outcome_detail says which.
    if d.get("status") != "verified":
        outcome = outcome_detail = "pending"
    else:
        said_yes = True if (d.get("kind") or "alert") == "alert" \
            else (d.get("probability") or 0.0) >= 0.5
        materialized = bool(d.get("materialized"))
        outcome = "HIT" if said_yes == materialized else "MISS"
        if said_yes and materialized:
            outcome_detail = "HIT"
        elif said_yes and not materialized:
            outcome_detail = "FALSE_ALARM"
        elif not said_yes and materialized:
            outcome_detail = "MISS"
        else:
            outcome_detail = "CORRECT_REJECTION"
    err, perr, cerr = d.get("err_dex"), d.get("persist_err_dex"), d.get("clim_err_dex")
    return {"id": d.get("id"), "kind": d.get("kind") or "alert", "issued_ct": _to_ct(d.get("created_at")),
            "forecast_threshold": d.get("threshold"), "p_alert": rd(d.get("probability")),
            "p_M_24h": rd(d.get("p_m")), "p_X_24h": rd(d.get("p_x")),
            "expected_class": d.get("expected_class"), "current_class_at_issue": d.get("current_class"),
            "window_h": d.get("window_h"), "verify_after_ct": _to_ct(d.get("verify_after")),
            "status": d.get("status"), "actual_peak_class": d.get("actual_peak_class"),
            "actual_peak_flux_wm2": d.get("actual_peak_flux"), "outcome": outcome,
            "outcome_detail": outcome_detail,
            "verified_ct": _to_ct(d.get("verified_at")), "alert_email": d.get("alert_status"),
            "followup": d.get("followup_status"),
            "predicted_peak_class": d.get("pred_peak_class"),
            "predicted_peak_flux_wm2": d.get("pred_peak_flux"),
            "err_dex": rd3(err), "persistence_err_dex": rd3(perr), "climatology_err_dex": rd3(cerr),
            "beat_persistence": _beat(err, perr), "beat_climatology": _beat(err, cerr)}


def history_csv_string(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    conn = _db(cfg)
    try:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id").fetchall()
    finally:
        conn.close()
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_CSV_COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(_row_record(r))
    return buf.getvalue()


def _history_key(row: dict) -> tuple:
    """(kind, issued_ct) — NOT id: each machine's SQLite `id` is a local
    autoincrement, so ids collide across machines. issued_ct (to-the-minute
    Central time) is effectively unique per forecast and IS shared meaning."""
    return (row.get("kind", ""), row.get("issued_ct", ""))


def _history_rank(row: dict) -> tuple:
    """Which of two duplicate rows to keep when merging: verified beats
    pending, then whichever has more filled-in fields (a fuller record)."""
    verified = 1 if row.get("status") == "verified" else 0
    filled = sum(1 for v in row.values() if v not in (None, ""))
    return (verified, filled)


def _merge_history_rows(local_rows: list[dict], disk_rows: list[dict]) -> list[dict]:
    """Union this machine's DB rows with whatever is already in the file on
    disk (e.g. pushed by a teammate's machine), keyed by (kind, issued_ct).
    A plain overwrite would silently delete every other machine's rows every
    time the local notifier exports — this merges instead."""
    merged: dict[tuple, dict] = {}
    for row in disk_rows + local_rows:            # local processed last: wins ties (freshest)
        k = _history_key(row)
        if k not in merged or _history_rank(row) >= _history_rank(merged[k]):
            merged[k] = row
    rows = sorted(merged.values(), key=lambda r: r.get("issued_ct") or "")
    for i, row in enumerate(rows, start=1):
        row["id"] = str(i)                          # renumber for display only
    return rows


def export_csv(cfg: dict, path: str | None = None) -> str | None:
    """Write the full prediction history to a CSV spreadsheet on disk (Excel/
    Sheets), MERGED with whatever is already at that path — never a blind
    overwrite, so other machines' rows (e.g. teammates', or a prior deploy's)
    survive this machine's export."""
    import csv
    import io
    path = path or (cfg.get("notify", {}).get("history_csv")
                    or os.path.join(cfg["_project_root"], "prediction_history.csv"))
    try:
        local_rows = list(csv.DictReader(io.StringIO(history_csv_string(cfg))))
        disk_rows = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                disk_rows = list(csv.DictReader(fh))
        merged = _merge_history_rows(local_rows, disk_rows)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_COLS, extrasaction="ignore")
            w.writeheader()
            for row in merged:
                w.writerow(row)
        return path
    except OSError:
        return None


def run_once(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    if not cfg.get("notify", {}).get("enabled", True):
        return {"enabled": False}
    conn = _db(cfg)
    try:
        alert = check_and_alert(cfg, conn)
        daily_log = log_daily_forecast(cfg, conn)            # one forecast row per day
        verifications = verify_pending(cfg, conn)            # fill in what actually happened
        daily = maybe_send_daily(cfg, conn)
    finally:
        conn.close()
    csv_path = export_csv(cfg)                                # keep the spreadsheet fresh
    return {"checked_at": _now().isoformat(), "alert": alert, "daily_log": daily_log,
            "verifications": verifications, "daily_digest": daily, "history_csv": csv_path}


def prospective_record(conn: sqlite3.Connection, cfg: dict) -> dict:
    """The PROSPECTIVE forecast record — the gold standard no retrospective test
    can match: every verified daily row was issued BEFORE the outcome was
    knowable. Scores the daily P(M+ in 24h) forecasts against what the Sun then
    did: Brier score (proper, works at any n) + TSS at the alert threshold once
    enough days accumulate. Grows automatically as the notifier runs."""
    rows = conn.execute(
        "SELECT probability, materialized FROM predictions "
        "WHERE kind='daily' AND status='verified' AND probability IS NOT NULL "
        "AND materialized IS NOT NULL ORDER BY id").fetchall()
    n = len(rows)
    if n == 0:
        return {"available": False, "n_days": 0,
                "note": "builds one verified forecast per day the notifier runs — "
                        "a true prospective test (forecast issued before the outcome)"}
    p = [float(r["probability"]) for r in rows]
    y = [int(r["materialized"]) for r in rows]
    brier = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n
    events = sum(y)
    thr = float(cfg.get("notify", {}).get("alert_probability", 0.5))
    tp = sum(1 for pi, yi in zip(p, y) if pi >= thr and yi)
    fp = sum(1 for pi, yi in zip(p, y) if pi >= thr and not yi)
    fn = sum(1 for pi, yi in zip(p, y) if pi < thr and yi)
    tn = sum(1 for pi, yi in zip(p, y) if pi < thr and not yi)
    tss = None
    if (tp + fn) and (fp + tn):                       # both classes observed
        tss = round(tp / (tp + fn) - fp / (fp + tn), 3)
    return {
        "available": True,
        "n_days": n,
        "events": events,                             # days an M+ flare actually followed
        "base_rate": round(events / n, 3),
        "brier": round(brier, 4),
        "brier_climatology": round(sum((events / n - yi) ** 2 for yi in y) / n, 4),
        "tss_at_alert_threshold": tss,                # None until both outcomes seen
        "alert_threshold_p": thr,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "note": "prospective — every forecast was issued before its outcome window; "
                "TSS appears once both flare and no-flare days exist",
    }


def status(cfg: dict | None = None, limit: int = 20) -> dict:
    cfg = cfg or load_config()
    conn = _db(cfg)
    try:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                            (limit,)).fetchall()
        pending = conn.execute("SELECT COUNT(*) c FROM predictions WHERE status='pending'").fetchone()["c"]
        total = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
        verified = conn.execute("SELECT COUNT(*) c FROM predictions WHERE status='verified'").fetchone()["c"]
        hits = conn.execute("SELECT COUNT(*) c FROM predictions WHERE materialized=1").fetchone()["c"]
        last_digest = _meta_get(conn, "last_digest_date")
        graded = conn.execute(
            "SELECT err_dex, persist_err_dex, clim_err_dex FROM predictions "
            "WHERE status='verified' AND err_dex IS NOT NULL ORDER BY id").fetchall()
        prospective = prospective_record(conn, cfg)
    finally:
        conn.close()
    from . import magnitude
    accuracy = magnitude.skill_summary([dict(g) for g in graded])
    n = cfg.get("notify", {})
    rec = _recipients(cfg)
    dry = not (n.get("smtp_host") and rec and os.environ.get("SMTP_PASSWORD"))
    return {
        "enabled": bool(n.get("enabled", True)),
        "mode": "dry-run" if dry else "live-email",
        "recipients": rec,
        "sender": n.get("sender") or n.get("smtp_user") or None,
        "threshold": n.get("alert_threshold", "M"),
        "alert_probability": n.get("alert_probability", 0.5),
        "observation_window_h": n.get("observation_window_h", 24),
        "daily_forecast": bool(n.get("daily_forecast_enabled", True)),
        "daily_send_hour_utc": n.get("daily_send_hour_utc", 13),
        "last_digest_date": last_digest,
        "pending": pending,
        "history": {"total": total, "verified": verified, "hits": hits,
                    "csv_url": "/api/notify/history.csv"},
        "accuracy": accuracy,            # honest peak-magnitude skill (log-flux vs baselines)
        "prospective": prospective,      # the growing real-time forecast record
        "recent": [dict(r) for r in rows],
    }


# ----------------------------------------------------------------------
# In-server background loop (started from the API lifespan; safe + dry-run)
# ----------------------------------------------------------------------
def start_background(cfg: dict | None = None) -> bool:
    """Start the notifier loop in a daemon thread (once). Returns True if started."""
    global _bg_started
    cfg = cfg or load_config()
    n = cfg.get("notify", {})
    if _bg_started or not (n.get("enabled", True) and n.get("run_in_server", True)):
        return False
    _bg_started = True
    interval = max(60, int(n.get("poll_interval_min", 30)) * 60)

    def _loop():
        import time
        while True:
            try:
                run_once(cfg)
            except Exception as exc:                          # noqa: BLE001
                log.warning("notifier cycle error: %s", type(exc).__name__)
            time.sleep(interval)
    threading.Thread(target=_loop, name="helios-notifier", daemon=True).start()
    log.info("notifier background loop started (every %d min, %s)",
             interval // 60, status(cfg)["mode"])
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Helios flare email notifier.")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    ap.add_argument("--status", action="store_true", help="print state and exit")
    args = ap.parse_args()
    cfg = load_config()
    if args.status:
        print(json.dumps(status(cfg), indent=2, default=str)); return
    if args.once:
        print(json.dumps(run_once(cfg), indent=2, default=str)); return
    import time
    interval = max(60, int(cfg["notify"].get("poll_interval_min", 30)) * 60)
    print(f"Helios notifier running every {interval//60} min "
          f"({status(cfg)['mode']} mode). Ctrl+C to stop.")
    while True:
        summary = run_once(cfg)
        a = summary.get("alert", {})
        print(f"[{summary.get('checked_at')}] alert={a.get('triggered')} "
              f"verifications={len(summary.get('verifications', []))}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
