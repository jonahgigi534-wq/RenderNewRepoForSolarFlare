"""Physics interpretation (PAPER.md section 10, Figures 5-6):
(1) which SHARP parameters drive the flare model (RandomForest feature importance,
    aggregated per parameter through the calibration/pipeline wrappers), and
(2) PCA of the SHARP feature space (variance structure + PC1 loadings).
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _common import ROOT, RESULTS, load_or_build

from solarflare import data as dataio
from solarflare.config import load_config

OUT = os.path.join(RESULTS, "physics.json")
cfg = load_config()
keys = cfg["sharp_live"]["keywords"]
stats = cfg["features"]["summary_stats"]

# ---- (1) Feature importance from the deployed live model ----
payload = joblib.load(os.path.join(ROOT, "models", "flare_sharp_live_model.joblib"))


def _rf_importances(est):
    """Reach the RandomForest's feature_importances_ through pipeline/calibration wrappers."""
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_)
    if hasattr(est, "steps"):                              # a Pipeline
        return _rf_importances(est.steps[-1][1])
    if hasattr(est, "calibrated_classifiers_"):            # CalibratedClassifierCV
        return np.mean([_rf_importances(cc.estimator) for cc in est.calibrated_classifiers_], axis=0)
    raise SystemExit(f"no feature_importances_ in {type(est).__name__}")


imp = _rf_importances(payload["model"])
imp = np.asarray(imp) / np.asarray(imp).sum()              # 119 = 17 keys x 7 stats
per_param = {k: float(imp[i * len(stats):(i + 1) * len(stats)].sum()) for i, k in enumerate(keys)}
per_param = dict(sorted(per_param.items(), key=lambda kv: -kv[1]))
print("=== Per-parameter importance (sum over its 7 summary stats) ===", flush=True)
for k, v in per_param.items():
    print(f"  {k:10s} {v:.4f}", flush=True)

# ---- (2) PCA of the SHARP feature space (sample of live JSOC) ----
d = load_or_build("pca_2014jan", datetime(2014, 1, 1, tzinfo=timezone.utc),
                  datetime(2014, 1, 22, tzinfo=timezone.utc), cfg)
Xf = dataio.build_matrix(d["X3d"], cfg)
Xf = np.nan_to_num(Xf, nan=np.nanmedian(Xf))
Xs = StandardScaler().fit_transform(Xf)
pca = PCA(random_state=42).fit(Xs)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
n_for_90 = int(np.argmax(cum >= 0.90) + 1)
print(f"\nPCA on {Xs.shape}: PC1 {evr[0]:.1%}, PC1-3 {cum[2]:.1%}, {n_for_90} PCs for 90%", flush=True)

load1 = np.abs(pca.components_[0])
pc1_param = {k: float(load1[i * len(stats):(i + 1) * len(stats)].sum()) for i, k in enumerate(keys)}
pc1_param = dict(sorted(pc1_param.items(), key=lambda kv: -kv[1]))
print("PC1 dominated by:", list(pc1_param)[:5], flush=True)

json.dump({
    "importance_per_param": per_param,
    "pca_explained_variance_ratio": [round(float(x), 4) for x in evr[:10]],
    "pca_cumulative": [round(float(x), 4) for x in cum[:10]],
    "pca_pc1_pct": round(float(evr[0]), 4),
    "pca_pc1to3_pct": round(float(cum[2]), 4),
    "pca_n_components_90pct": n_for_90,
    "pca_pc1_top_params": list(pc1_param)[:5],
    "pca_sample": {"n": int(Xs.shape[0])},
}, open(OUT, "w"), indent=2)
print(f"saved -> {OUT}", flush=True)
