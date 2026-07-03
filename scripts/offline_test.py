"""Offline CI gate — no network, no datasets, no model loads.

Three cheap layers that catch what auto-deploying main most needs caught:
  1. every module compiles (syntax / import-time NameErrors);
  2. the pure-math core gives known answers (TSS, label gate, magnitude,
     history merge);
  3. the committed research artifacts keep their JSON contract (the dashboard
     reads them directly).

The full suites (scripts/regression_test.py, scripts/smoke_test.py) need live
NOAA/JSOC and stay a local-machine step. Run:  python scripts/offline_test.py
"""
from __future__ import annotations

import compileall
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# ---- 1. compile everything -------------------------------------------------
ok = all(compileall.compile_dir(os.path.join(ROOT, d), quiet=1, force=False)
         for d in ("solarflare", "api", "scripts"))
check("all modules compile", bool(ok))

# ---- 2. pure-math known answers ---------------------------------------------
import numpy as np                                        # noqa: E402

from solarflare.scorecard import (detect_operational_years,               # noqa: E402
                                  frozen_tss, label_excluded_years,
                                  label_gate_status, peak_tss)

y = np.array([1, 0, 1, 0])
p_perfect = np.array([0.9, 0.1, 0.8, 0.2])
check("peak TSS = 1 on a perfect ranking", abs(peak_tss(y, p_perfect) - 1.0) < 1e-9)
check("frozen TSS = 1 at a separating threshold", abs(frozen_tss(y, p_perfect, 0.5) - 1.0) < 1e-9)
check("frozen TSS = -1 when inverted", abs(frozen_tss(y, 1 - p_perfect, 0.5) + 1.0) < 1e-9)
check("peak TSS clips to 0 on anti-skill", peak_tss(y, 1 - p_perfect) == 0.0)
check("degenerate one-class set scores 0", peak_tss(np.zeros(4), p_perfect) == 0.0)

cfg_gate = {"scorecard": {"min_label_attribution": 0.8,
                          "label_attribution_by_year": {2014: 0.94, 2015: 0.98,
                                                        2023: 0.15}}}
check("label gate flags 2023 only", label_excluded_years(cfg_gate) == {2023: 0.15})
check("label gate empty without cfg", label_excluded_years(None) == {})
check("gate passes a measured-good year", label_gate_status(cfg_gate, 2015)[0])
check("gate rejects a measured-bad year", not label_gate_status(cfg_gate, 2023)[0])
check("gate is FAIL-CLOSED for unmeasured years",
      not label_gate_status(cfg_gate, 2016)[0]
      and "unmeasured" in label_gate_status(cfg_gate, 2016)[1])
d = tempfile.mkdtemp()
for yr in (2015, 2016, 2023, 2014):
    open(os.path.join(d, f"dataset_{yr}.npz"), "w").close()
check("ungated year detection keeps 2023 (noaa_baseline path)",
      detect_operational_years(d) == [2015, 2016, 2023])
check("gated detection drops measured-bad 2023 AND unmeasured 2016",
      detect_operational_years(d, cfg_gate) == [2015])

from solarflare import magnitude                          # noqa: E402
from solarflare.config import load_config                 # noqa: E402

cfg = load_config()
e = magnitude.error_dex(1.3e-6, 9.5e-6)
check("log-flux error sane (C1.3 vs C9.5 ~0.86 dex)", e is not None and 0.8 < e < 0.95, str(e))
check("model peak rises with M+ odds",
      magnitude.expected_peak_flux(0.95, 0.7, 0.2, cfg)
      > magnitude.expected_peak_flux(0.3, 0.05, 0.005, cfg))

from solarflare import notify                             # noqa: E402

row = {c: "" for c in notify._CSV_COLS}
row.update(kind="alert", issued_ct="1999-01-01 00:00 CDT", status="pending")
merged = notify._merge_history_rows([], [row])
check("history merge keeps a disk-only row",
      any(r["issued_ct"] == "1999-01-01 00:00 CDT" for r in merged))
merged2 = notify._merge_history_rows([{**row, "status": "verified", "outcome": "HIT"}], [row])
check("history merge prefers verified over pending", merged2[0]["status"] == "verified")

# ---- 3. committed-artifact contracts ----------------------------------------
def _load(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

sc = _load("skill_scorecard.json")
if sc is not None:
    ms = sc.get("models") or []
    check("scorecard has model rows", bool(ms), str(len(ms)))
    check("scorecard rows carry frozen blocks", all(isinstance(m.get("frozen"), dict) for m in ms))
    check("scorecard findings carry frozen gap + CI",
          sc.get("findings", {}).get("frozen_overstatement_gap") is not None
          and isinstance(sc.get("findings", {}).get("frozen_overstatement_gap_ci"), list))
dr = _load("dose_response.json")
if dr is not None:
    check("dose_response steps present", bool(dr.get("steps")), str(len(dr.get("steps", []))))
st = _load("storm_scorecard.json")
if st is not None:
    check("storm scorecard carries peak+frozen", "peak" in st and "frozen" in st)

print()
if failures:
    print(f"OFFLINE CI FAILED: {failures}")
    sys.exit(1)
print("ALL OFFLINE CHECKS PASSED.")
