"""Run rolling-window GARCH(1,1) baselines for all (index, period, dist) combos.

Saves one CSV per combo to results/var_forecasts/garch_{index}_{period}_{dist}.csv.
Each CSV has columns: date, return, var_forecast, exceeded.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import get_window
from src.garch_baseline import rolling_garch_var

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "var_forecasts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDICES = ["wig20", "spx", "ftse"]
PERIODS = {
    "2009": "2005-01-01",
    "2011": "2007-01-01",
    "2017": "2013-01-01",
    "2020": "2016-01-01",
}
DISTS = ["normal", "t", "skewt"]
ALPHA = 0.025
N_TRAIN = 1000
N_TEST = 250


def run_one(index: str, period_label: str, start_date: str, dist: str) -> Path:
    out_path = OUT_DIR / f"garch_{index}_{period_label}_{dist}.csv"
    if out_path.exists():
        print(f"    SKIP (exists): {out_path.name}")
        return out_path

    df = pd.read_parquet(PROCESSED_DIR / f"{index}_returns.parquet")
    train, test, dates = get_window(df, start_date, n_train=N_TRAIN, n_test=N_TEST)

    # rolling_garch_var refits at every step on a sliding window of length n_train
    # We pass the full (n_train + n_test) slice and let the function do the windowing.
    full_returns = np.concatenate([train, test])
    t0 = time.time()
    var_fc = rolling_garch_var(full_returns, n_train=N_TRAIN, n_test=N_TEST,
                                dist=dist, alpha=ALPHA, verbose=False)
    elapsed = time.time() - t0

    exceeded = (test < var_fc).astype(int)
    df_out = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "return": test,
        "var_forecast": var_fc,
        "exceeded": exceeded,
    })
    df_out.to_csv(out_path, index=False)
    n_excep = int(exceeded.sum())
    print(f"    DONE in {elapsed:5.1f}s  exceptions={n_excep:3d}/{N_TEST}  -> {out_path.name}")
    return out_path


if __name__ == "__main__":
    total = len(INDICES) * len(PERIODS) * len(DISTS)
    counter = 0
    t_start = time.time()
    for index in INDICES:
        for period_label, start_date in PERIODS.items():
            for dist in DISTS:
                counter += 1
                print(f"[{counter:2d}/{total}] {index} period={period_label} dist={dist}")
                try:
                    run_one(index, period_label, start_date, dist)
                except Exception as e:
                    print(f"    FAILED: {e}")
    total_elapsed = time.time() - t_start
    print(f"\nTotal: {total_elapsed/60:.1f} min")