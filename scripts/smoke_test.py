"""End-to-end smoke test — proves the whole pipeline runs without the big
dataset download or a trained model. Exercises:

  1. feature engineering on synthetic SWAN-SF-shaped data,
  2. training + threshold tuning + skill-score evaluation,
  3. saving and reloading the .joblib/.pkl artifacts,
  4. live NOAA fetch -> nowcast -> flux forecast -> ensemble.

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solarflare import predictor, train
from solarflare.config import load_config


def main() -> int:
    # Train into a TEMP model dir so the smoke test never clobbers a real
    # trained model sitting in models/.
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["model_dir"] = tempfile.mkdtemp(prefix="helios_smoke_")
    print("=" * 60)
    print("1) TRAIN on synthetic fixture  (temp dir, real model untouched)")
    print("=" * 60)
    payload = train.train(synthetic=True, cfg=cfg)
    tss = payload["metrics"]["tss"]
    assert tss > 0.3, f"expected real skill on synthetic data, got TSS={tss}"
    print(f"OK  synthetic TSS={tss:.3f} (>0.3)")

    print("\n" + "=" * 60)
    print("2) RELOAD model + run SHARP forecast on one sample")
    print("=" * 60)
    predictor._MODEL_CACHE = None
    model = predictor.load_sharp_model(cfg)
    assert model is not None, "model failed to reload"
    import numpy as np
    n_feat = len(payload["feature_names"])
    fc = predictor.sharp_forecast(np.zeros(n_feat), cfg)
    assert fc and fc["available"]
    print(f"OK  SHARP forecast: p(M+ 24h)={fc['p_M_or_greater_24h']}  "
          f"-> {fc['prediction']}")

    print("\n" + "=" * 60)
    print("3) LIVE end-to-end predict() (NOAA fetch may be live or cached)")
    print("=" * 60)
    result = predictor.predict(cfg=cfg)
    print(f"data status : {result['data_freshness']['status']}")
    if result["nowcast"].get("available"):
        nc = result["nowcast"]
        print(f"now flaring : {nc['is_flaring']}  current={nc['current_class']}  "
              f"peak24h={nc['peak_24h_class']}")
        print(f"X-warning   : {nc['x_warning']['level']}")
    fc24 = result["forecast"]["flux_track"]
    if fc24.get("available"):
        h = fc24["horizons"]["24h"]
        print(f"24h flux fc : p(M+)={h['p_M_or_greater']}  p(X)={h['p_X_class']}")
    assert "ensemble_24h" in result["forecast"]
    print("OK  full prediction assembled.")

    print("\nALL SMOKE TESTS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
