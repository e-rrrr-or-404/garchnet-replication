# Project Status

**Paper being replicated:** Buczynski & Chlebus (2024), "GARCHNet: Value-at-Risk
Forecasting with GARCH Models Based on Neural Networks," *Computational Economics*.

**Presentation date:** Week of July 6, 2026.

**One-line summary:** We replicate the paper's three GARCH baselines (normal,
Student-t, Hansen skewed-t innovations) and the proposed GARCHNet model (LSTM
parameterizes the conditional variance, trained by exact NLL of the assumed
innovation distribution). We compare them on Value-at-Risk forecasts for WIG20,
S&P 500, and FTSE 100 across four time periods (test years 2009, 2011, 2017, 2020).

---

## What is done

### Data layer (DONE)
- Daily closes for WIG20, S&P 500, FTSE 100, 2004-01-01 to 2021-12-31.
- Sources: WIG20 via manual Stooq CSV (Yahoo doesn't carry WIG20 reliably).
  SPX (^GSPC) and FTSE (^FTSE) via yfinance. All frozen into `data/raw/` and
  `data/processed/` in the repo for reproducibility.
- Log returns computed; sample stats sanity-checked against published daily vols
  for each index (SPX ~1.2%, FTSE ~1.1%, WIG20 ~1.4%).

### GARCH baselines (DONE)
- GARCH(1,1) with three innovation distributions, fit via the `arch` package.
- 250-day rolling-window VaR forecasts at alpha = 2.5%.
- Hansen (1994) skewed-t implemented by hand in `src/skewed_t.py` with passing
  sanity tests (PDF integrates to 1, zero mean / unit variance, MLE recovers
  parameters from synthetic data).
- 36 combos completed (3 indices × 4 periods × 3 distributions) saved as CSVs
  in `results/var_forecasts/garch_*.csv`. Runtime: 7 minutes total.
- Exception counts: see `notebooks/02_garch_baseline.ipynb` or run
  `python -c "..."` from `STATUS.md` section "Useful one-liners."

### Replication notes
- Our exception counts are systematically *lower* than the paper's in turbulent
  periods (2009, 2020). Likely cause: the paper does not publish exact training
  window dates. Our training windows include the September 2008 and March 2020
  crash days, which inflates conditional sigma and widens the VaR bands. If the
  paper's training windows ended before these shocks, their narrower bands would
  produce more exceptions on the same test days. Document this honestly in the
  slides; do not pretend to match their numbers exactly.

---

## What still needs to be done

### Critical path (must finish by July 5)

1. **GARCHNet model** (`src/garchnet.py`) — biggest single piece
   - PyTorch LSTM(100) → FC(64) → FC(32) → output heads (sigma², eta, lambda).
   - Differentiable PyTorch NLL losses for normal, t, skewed-t.
   - Verify gradients flow through the skewed-t loss (most error-prone piece).

2. **GARCHNet rolling-window driver** (`src/run_garchnet.py`)
   - Same structure as `src/run_garch_baselines.py`.
   - Trains a fresh net for each of 250 forecasts in the test window.
   - Runs on Colab GPU; uploads CSVs back to the repo via Google Drive sync
     or manual download + push.
   - 3 seeds per combo for variance estimates beyond the paper.

3. **Backtesting tests** (`src/backtest.py`)
   - Kupiec UC, Christoffersen CC, Engle-Manganelli DQ.
   - Loss functions: Lopez LLF, Caporin CRLF / CFLF, Abad-Benito-Lopez ABLLF.
   - Gneiting (2011) GPL scoring function.
   - Each test takes a (returns, var_forecasts) pair and returns a p-value or score.

4. **Results compilation** (`notebooks/04_results_tables.ipynb`)
   - Reproduces the paper's Tables 1–6 format with our numbers.
   - Side-by-side GARCH vs GARCHNet for each (index, period, distribution).

### Nice-to-have (only if time permits)

5. **Extension experiment** — one of:
   - GJR-GARCHNet (asymmetric volatility shocks)
   - GRU or Transformer in place of LSTM
   - Apply to NIFTY 50 / SENSEX (Indian indices not in the paper)
   - Add early stopping inside the training loop

6. **Slides** (Google Slides or PowerPoint, ~15 slides)
   - Outlined in `notes/presentation_outline.md`.

---

## Useful one-liners

Check data is loaded:
python -c "import pandas as pd; [print(name, pd.read_parquet(f'data/processed/{name}_returns.parquet').shape) for name in ['wig20','spx','ftse']]"

Check GARCH baseline exception counts:
python -c "

import pandas as pd

from pathlib import Path

files = sorted(Path('results/var_forecasts').glob('garch_*.csv'))

rows = []

for f in files:

parts = f.stem.split('_')

df = pd.read_csv(f)

rows.append({'index': parts[1], 'period': parts[2], 'dist': parts[3], 'exceptions': int(df['exceeded'].sum())})

print(pd.DataFrame(rows).pivot_table(index=['index','period'], columns='dist', values='exceptions'))

"

Run the skewed-t sanity tests:
python tests\test_skewed_t.py

---

## What to read first

1. The paper itself: in the email Ira forwarded. Skim Section 2 (methodology) and
   Section 3 (data). Section 4 (results) gives you tables to match.
2. `src/skewed_t.py` — the only math we coded by hand. Make sure you can explain
   eq. 8 and 9 from the paper.
3. `src/garch_baseline.py` — wraps the `arch` package. Be ready to explain
   what `arch_model(..., dist='skewt')` actually optimizes.

---

## Known limitations (mention in slides)

- Exact training/test window dates are unspecified in the paper; ours land
  within a few weeks of the paper's stated test years but don't match day-for-day.
- WIG20 data is from Stooq (matches paper); SPX and FTSE are from Yahoo (paper
  also uses Stooq, but Yahoo and Stooq differ by <0.01% on daily closes for
  these indices).
- We will run 3 random seeds per GARCHNet config; the paper runs 1.
- Compute constraints may force us to reduce p ∈ {5, 10, 20, 100} grid to
  {5, 20} only. The paper recommends p=20 as best.