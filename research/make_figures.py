"""Regenerate all board/paper figures (figures/fig1-6) from research/results/*.json."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import ROOT, RESULTS, FIGURES

plt.rcParams.update({"figure.dpi": 140, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})
C_BENCH, C_LIVE, C_HR, C_BAD = "#3b5bdb", "#e8590c", "#f08c00", "#c92a2a"

mp = json.load(open(os.path.join(RESULTS, "multiperiod_rescore.json")))
sc = json.load(open(os.path.join(ROOT, "skill_scorecard.json")))
ks = json.load(open(os.path.join(RESULTS, "exp2_distribution.json")))
ph = json.load(open(os.path.join(RESULTS, "physics.json")))
BENCH_TSS = mp["benchmark_swansf"]["tss"]

# Pretty labels per period tag; figures derive their period lists from the JSON
# artifacts, so label-gated exclusions (e.g. 2023) drop out automatically.
PERIOD_LABELS = {
    "2014_solar_max": "2014\n(solar max)",
    "2015_declining": "2015\n(declining)",
    "2017_declining": "2017\n(declining, Sept X-flares)",
    "2023_rising_max": "2023\n(rising max)",
}
plabel = lambda tag: PERIOD_LABELS.get(tag, tag)

def excluded_note(d, key):
    ex = d.get(key) or {}
    return "; ".join(f"{tag.split('_')[0]} excluded: label-attribution "
                     f"{'unmeasured' if 'unmeasured' in v['reason'] else 'below gate'}"
                     for tag, v in sorted(ex.items()))

# ---------- Fig 1: Exp 1 multi-period ----------
periods = [(tag, plabel(tag)) for tag in mp["periods"]]
labels = [p[1] for p in periods]
bal = [mp["periods"][p[0]]["by_operating_point"]["balanced"] for p in periods]
hr = [mp["periods"][p[0]]["by_operating_point"]["high_recall"] for p in periods]

def err(rows):
    lo = [r["tss"] - r["tss_ci95"][0] for r in rows]
    hi = [r["tss_ci95"][1] - r["tss"] for r in rows]
    return [lo, hi]

x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.axhline(BENCH_TSS, ls="--", color=C_BENCH, lw=2, label=f"SWAN-SF benchmark ({BENCH_TSS})")
ax.bar(x - w/2, [r["tss"] for r in bal], w, yerr=err(bal), capsize=4, color=C_LIVE, label="live @ default")
ax.bar(x + w/2, [r["tss"] for r in hr], w, yerr=err(hr), capsize=4, color=C_HR, label="live @ high-recall")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("TSS (True Skill Statistic)")
ax.set_ylim(0, 1)
ax.set_title("Benchmark sits above live default-threshold skill in every period")
ax.legend(loc="upper right", framealpha=0.95)
fig.tight_layout()
note = excluded_note(mp, "excluded_periods")
if note:
    fig.subplots_adjust(bottom=0.17)
    fig.text(0.99, 0.02, note, ha="right", fontsize=8, color="#495057")
fig.savefig(os.path.join(FIGURES, "fig1_multiperiod.png")); plt.close(fig)

# ---------- Fig 2: Exp 3 the 2x2 scorecard ----------
# Models without a leakage-free benchmark cell (benchmark_tss null) are plotted
# operational-only in the footnote, not as bars.
models = [m for m in sc["models"] if m.get("benchmark_tss") is not None]
excluded = [m for m in sc["models"] if m.get("benchmark_tss") is None]
names = [m["name"].replace(" (multi-year)", "\n(multi-year)") for m in models]
btss = [m["benchmark_tss"] for m in models]; otss = [m["operational_tss"] for m in models]
n_years = len(models[0].get("operational_years_used", [])) or 1
op_label = f"operational test ({n_years}-year mean)" if n_years > 1 else "operational test (live)"
x = np.arange(len(names)); w = 0.36
fig, ax = plt.subplots(figsize=(7.6, 4.6))
b1 = ax.bar(x - w/2, btss, w, color=C_BENCH, label="benchmark test")
b2 = ax.bar(x + w/2, otss, w, color=C_LIVE, label=op_label)
for i, m in enumerate(models):
    ax.annotate(f"gap {m['gap']:+.3f}", (x[i], 1.06), ha="center",
                color=C_BAD, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel(sc.get("metric", "TSS").split(" (")[0])
ax.set_ylim(0, 1.16)
ax.set_title("Retraining on live data does not close the gap\n"
             "(single-year widens it; multi-year cell not leakage-free, §8)")
ax.legend(loc="lower center"); ax.bar_label(b1, fmt="%.2f", padding=2); ax.bar_label(b2, fmt="%.2f", padding=2)
if excluded:
    note = "; ".join(f"{m['name']}: operational TSS {m['operational_tss']:.2f} "
                     "(no leakage-free benchmark test)" for m in excluded)
    fig.text(0.99, 0.005, note, ha="right", fontsize=8, color="#495057")
fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig2_scorecard_2x2.png")); plt.close(fig)

# ---------- Fig 3: Exp 2 KS distribution shift ----------
feats = sorted(ks["features"].items(), key=lambda kv: kv[1]["ks_D"])
names = [f[0] for f in feats]; D = [f[1]["ks_D"] for f in feats]
fig, ax = plt.subplots(figsize=(7.2, 5.2))
colors = [C_BAD if d >= 0.2 else C_LIVE if d >= 0.1 else C_HR for d in D]
ax.barh(names, D, color=colors)
ax.set_xlabel("Kolmogorov–Smirnov D  (benchmark era 2011 vs operational 2015)")
ax.set_title(f"All 17/17 SHARP features shift (median D = {ks['median_ks_D']})")
ax.axvline(ks["median_ks_D"], ls="--", color="#495057", lw=1.2, label=f"median {ks['median_ks_D']}")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig3_distribution_shift.png")); plt.close(fig)

# ---------- Fig 4: accuracy illusion (2014 numbers) ----------
p2014 = mp["periods"]["2014_solar_max"]; n = p2014["n"]; pos = p2014["positives"]
base = pos / n
hrr = p2014["by_operating_point"]["high_recall"]
tp = round(hrr["recall"] * pos); fn = pos - tp
fp = round(tp * (1 - hrr["precision"]) / hrr["precision"]); tn = (n - pos) - fp
acc_real = (tp + tn) / n
acc_null = 1 - base
groups = ["Always \"no-flare\"\n(zero skill)", "Real model\n@ high-recall"]
accs = [acc_null, acc_real]; tsss = [0.0, hrr["tss"]]
x = np.arange(len(groups)); w = 0.36
fig, ax = plt.subplots(figsize=(7.2, 4.4))
b1 = ax.bar(x - w/2, accs, w, color="#adb5bd", label="Accuracy")
b2 = ax.bar(x + w/2, tsss, w, color=C_LIVE, label="TSS (real skill)")
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("score"); ax.set_ylim(0, 1.05)
ax.set_title("The accuracy illusion: a zero-skill model 'wins' on accuracy")
ax.bar_label(b1, fmt="%.2f", padding=2); ax.bar_label(b2, fmt="%.2f", padding=2)
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig4_accuracy_illusion.png")); plt.close(fig)

# ---------- Fig 5: feature importance ----------
imp = ph["importance_per_param"]
names = list(imp)[::-1]; vals = [imp[k] for k in names]
top5 = set(list(imp)[:5])
colors = ["#c92a2a" if k in top5 else "#f08c00" for k in names]
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.barh(names, vals, color=colors)
ax.set_xlabel("Feature importance (RandomForest, summed over 7 summary stats)")
ax.set_title("What drives the flare model: extensive current/flux parameters")
ax.annotate("top 5 = 72% of the model's\ndecisions (red)", (max(vals) * 0.55, 2.2),
            color="#c92a2a", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig5_feature_importance.png")); plt.close(fig)

# ---------- Fig 6: PCA scree ----------
evr = ph["pca_explained_variance_ratio"]; cum = ph["pca_cumulative"]
n90 = ph["pca_n_components_90pct"]
x = np.arange(1, len(evr) + 1)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.bar(x, [v * 100 for v in evr], color=C_BENCH, label="per-PC variance")
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(x, [c * 100 for c in cum], "o-", color=C_LIVE, label="cumulative")
ax.set_xlabel("Principal component"); ax.set_ylabel("variance explained (%)")
ax2.set_ylabel("cumulative variance (%)"); ax2.set_ylim(0, 100)
ax.set_title(f"PCA of SHARP features: PC1 = {evr[0]:.0%} (AR size/energy axis); "
             f"{n90} PCs reach 90%")
ax.set_xticks(x)
l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, n1 + n2, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig6_pca_scree.png")); plt.close(fig)

# ---------- Fig 7: Exp 4 — the fix decomposed (threshold objective, not live data) ----------
rc_path = os.path.join(RESULTS, "recalibration.json")
if os.path.exists(rc_path):
    rc = json.load(open(rc_path))
    tests = [(tag, f"{tag.split('_')[0]} (unseen)") for tag in rc["tests"]]
    labels = [t[1] for t in tests]
    arms = [("default", f"default {rc['default_threshold']} (val-F1)", "#adb5bd"),
            ("benchmark_val_tss",
             f"benchmark-val TSS {rc.get('benchmark_val_tss_threshold', '')} (no live data)",
             C_BENCH),
            ("recalibrated",
             f"live-recalibrated {rc['recalibrated_threshold']} (frozen on 2014)", C_LIVE)]
    arms = [(k, lab, c) for k, lab, c in arms if k in rc["tests"][tests[0][0]]]
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for j, (key, lab, color) in enumerate(arms):
        rows = [rc["tests"][t[0]][key] for t in tests]
        b = ax.bar(x + (j - 1) * w, [r["tss"] for r in rows], w, yerr=err(rows),
                   capsize=3, color=color, label=lab)
        ax.bar_label(b, fmt="%.2f", padding=2, fontsize=9)
    for i, (t, _) in enumerate(tests):
        g = rc["tests"][t]["tss_gain"]; ci = rc["tests"][t].get("tss_gain_ci95")
        txt = f"paired gain +{g:.2f}" + (f"\n[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "")
        ax.annotate(txt, (x[i], 1.02), ha="center", color="#2b8a3e",
                    fontweight="bold", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("live TSS")
    ax.set_ylim(0, 1.22)
    ax.set_title("The fix is the threshold objective (TSS, not F1) — live data adds ≤ +0.02:\n"
                 "the benchmark's own val-TSS threshold ≈ the live-recalibrated one")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig7_recalibration_fix.png")); plt.close(fig)
    print("wrote fig7")

# ---------- Fig 8: calibration / reliability of the deployed model ----------
# "When we say 30%, does it happen ~30% of the time?" Reads the pooled-operational
# reliability bins already in skill_scorecard.json (regenerates with the pipeline),
# and reports the Murphy calibration-refinement decomposition of the Brier score:
#   Brier = Reliability - Resolution + Uncertainty   (from the binned data).
DEPLOYED = "Benchmark-trained"                       # the deployed flare model (MODEL_CARD §1)
rel = (sc.get("reliability") or {})
if rel:
    def murphy(binz, obase):
        N = sum(b["n"] for b in binz) or 1
        relc = sum(b["n"] * (b["p_mean"] - b["obs_freq"]) ** 2 for b in binz) / N
        res = sum(b["n"] * (b["obs_freq"] - obase) ** 2 for b in binz) / N
        return relc, res, obase * (1 - obase)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#495057", lw=1.4, label="perfect calibration")
    palette = {DEPLOYED: C_BENCH, "Live-trained": C_LIVE, "Live-trained (multi-year)": "#2b8a3e"}
    for name, r in rel.items():
        binz = [b for b in r.get("bins", []) if b.get("n")]
        if not binz:
            continue
        xs = [b["p_mean"] for b in binz]; ys = [b["obs_freq"] for b in binz]
        nmax = max(b["n"] for b in binz) ** 0.5
        sizes = [18 + 90 * (b["n"] ** 0.5) / nmax for b in binz]
        c = palette.get(name, "#868e96"); lw = 2.6 if name == DEPLOYED else 1.4
        # Line only through well-populated bins (n>=10) so singleton tail bins
        # (n=1, obs_freq 0/1) don't yank the curve; every bin still shows as a dot
        # sized by count, so sparse ones read honestly as low-confidence.
        solid = [b for b in binz if b["n"] >= 10]
        if len(solid) >= 2:
            ax.plot([b["p_mean"] for b in solid], [b["obs_freq"] for b in solid],
                    "-", color=c, lw=lw, alpha=0.9 if name == DEPLOYED else 0.5)
        ax.scatter(xs, ys, s=sizes, color=c, alpha=0.85, zorder=3,
                   label=f"{name} (Brier {r.get('brier', '—')})", edgecolor="white", linewidth=0.6)
    dep = rel.get(DEPLOYED)
    if dep and [b for b in dep.get("bins", []) if b.get("n")]:
        binz = [b for b in dep["bins"] if b.get("n")]
        relc, res, unc = murphy(binz, dep["base_rate"])
        ax.text(0.03, 0.97,
                f"Deployed model — Brier decomposition\n"
                f"reliability  {relc:.4f}  (miscalibration, ↓ better)\n"
                f"resolution   {res:.4f}  (discrimination, ↑ better)\n"
                f"uncertainty  {unc:.4f}  (base rate {dep['base_rate']:.3f})\n"
                f"Brier ≈ rel − res + unc = {relc - res + unc:.4f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                family="monospace", bbox=dict(boxstyle="round", fc="#f1f3f5", ec="#ced4da"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("forecast probability P(M+ in 24 h)")
    ax.set_ylabel("observed flare frequency")
    ax.set_title("Calibration on operational data: forecasts track reality\n"
                 "(dot size ∝ √count; pooled operational years)")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig8_calibration.png")); plt.close(fig)
    print("wrote fig8")

print("wrote figures to", FIGURES)
