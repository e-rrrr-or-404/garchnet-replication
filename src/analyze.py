"""Compute exception counts and backtesting statistics for all VaR forecasts.

Outputs:
  results/analysis_per_file.csv  -- one row per CSV (90 total)
  results/analysis_summary.csv   -- GARCHNet averaged over seeds (54 -> 18 rows)
  results/comparison_table.csv   -- side-by-side GARCH vs GARCHNet
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
FORECAST_DIR = ROOT / "results" / "var_forecasts"
OUT_DIR = ROOT / "results"
ALPHA = 0.025


# ---------- backtesting tests ----------

def kupiec_uc(n_excep, n, alpha=ALPHA):
    """Unconditional coverage LR test (Kupiec 1995). Returns p-value."""
    if n_excep == 0:
        lr = -2 * n * np.log(1 - alpha)
    elif n_excep == n:
        lr = -2 * n * np.log(alpha)
    else:
        p_hat = n_excep / n
        lr = -2 * (n_excep * np.log(alpha) + (n - n_excep) * np.log(1 - alpha)
                   - n_excep * np.log(p_hat) - (n - n_excep) * np.log(1 - p_hat))
    return 1 - stats.chi2.cdf(lr, df=1)


def christoffersen_ind(exceeded):
    """Independence LR test (first-order Markov). Returns p-value."""
    x = exceeded.values if hasattr(exceeded, 'values') else np.asarray(exceeded)
    n00 = n01 = n10 = n11 = 0
    for i in range(len(x) - 1):
        if x[i] == 0 and x[i+1] == 0: n00 += 1
        elif x[i] == 0 and x[i+1] == 1: n01 += 1
        elif x[i] == 1 and x[i+1] == 0: n10 += 1
        elif x[i] == 1 and x[i+1] == 1: n11 += 1
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return np.nan
    pi = (n01 + n11) / (n0 + n1)
    pi0 = n01 / n0
    pi1 = n11 / n1
    if pi == 0 or pi == 1 or pi0 in (0, 1) or pi1 in (0, 1):
        return 1.0
    lr_uncond = (n01 + n11) * np.log(pi) + (n00 + n10) * np.log(1 - pi)
    lr_cond = (n01*np.log(pi0) + n00*np.log(1-pi0)
               + n11*np.log(pi1) + n10*np.log(1-pi1))
    lr = -2 * (lr_uncond - lr_cond)
    return 1 - stats.chi2.cdf(lr, df=1)


def christoffersen_cc(n_excep, n, exceeded, alpha=ALPHA):
    """Conditional coverage = UC + Ind (df=2)."""
    if n_excep == 0:
        lr_uc = -2 * n * np.log(1 - alpha)
    elif n_excep == n:
        lr_uc = -2 * n * np.log(alpha)
    else:
        p_hat = n_excep / n
        lr_uc = -2 * (n_excep * np.log(alpha) + (n - n_excep) * np.log(1 - alpha)
                      - n_excep * np.log(p_hat) - (n - n_excep) * np.log(1 - p_hat))
    x = exceeded.values if hasattr(exceeded, 'values') else np.asarray(exceeded)
    n00 = n01 = n10 = n11 = 0
    for i in range(len(x) - 1):
        if x[i] == 0 and x[i+1] == 0: n00 += 1
        elif x[i] == 0 and x[i+1] == 1: n01 += 1
        elif x[i] == 1 and x[i+1] == 0: n10 += 1
        elif x[i] == 1 and x[i+1] == 1: n11 += 1
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return 1 - stats.chi2.cdf(lr_uc, df=2)
    pi = (n01 + n11) / (n0 + n1)
    pi0 = n01 / n0
    pi1 = n11 / n1
    if pi == 0 or pi == 1 or pi0 in (0, 1) or pi1 in (0, 1):
        lr_ind = 0.0
    else:
        lr_uncond = (n01 + n11) * np.log(pi) + (n00 + n10) * np.log(1 - pi)
        lr_cond = (n01*np.log(pi0) + n00*np.log(1-pi0)
                   + n11*np.log(pi1) + n10*np.log(1-pi1))
        lr_ind = -2 * (lr_uncond - lr_cond)
    lr_cc = lr_uc + lr_ind
    return 1 - stats.chi2.cdf(lr_cc, df=2)


def dq_test(returns, var_forecasts, exceeded, alpha=ALPHA, lags=4):
    """Dynamic Quantile test (Engle & Manganelli 2004).
    Tests that hits (centered exceedances) are uncorrelated with their lags.
    """
    hits = exceeded.values - alpha if hasattr(exceeded, 'values') else np.asarray(exceeded) - alpha
    var_f = var_forecasts.values if hasattr(var_forecasts, 'values') else np.asarray(var_forecasts)
    n = len(hits)
    # Regression: hits[t] on intercept + hits[t-1..t-lags] + VaR[t]
    X = np.ones((n - lags, 1 + lags + 1))
    for i in range(lags):
        X[:, 1 + i] = hits[lags - 1 - i : n - 1 - i]
    X[:, -1] = var_f[lags:]
    y = hits[lags:]
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        # DQ statistic
        dq_stat = beta @ X.T @ X @ beta / (alpha * (1 - alpha))
        return 1 - stats.chi2.cdf(dq_stat, df=X.shape[1])
    except np.linalg.LinAlgError:
        return np.nan

def loss_functions(returns, var_forecasts, exceeded):
    """Lopez quadratic and Caporin firm/regulator loss functions."""
    r = returns.values if hasattr(returns, 'values') else returns
    v = var_forecasts.values if hasattr(var_forecasts, 'values') else var_forecasts
    e = exceeded.values if hasattr(exceeded, 'values') else exceeded
    # Lopez quadratic: 1 + (VaR - r)^2 when r < VaR else 0
    lopez = np.where(e == 1, 1 + (v - r)**2, 0).sum()
    # Caporin regulator: 1 - r/VaR when r < VaR else 0 (loss to regulator)
    caporin_reg = np.where(e == 1, 1 - r/v, 0).sum()
    # Caporin firm: 1 - r/VaR for ALL days (opportunity cost of holding capital)
    caporin_firm = (1 - r/v).sum()
    return {'lopez': lopez, 'caporin_reg': caporin_reg, 'caporin_firm': caporin_firm}


# ---------- per-file analysis ----------

def analyze_file(path):
    df = pd.read_csv(path)
    n = len(df)
    n_excep = int(df['exceeded'].sum())
    parts = path.stem.split('_')
    model = parts[0]
    losses = loss_functions(df['return'], df['var_forecast'], df['exceeded'])
    index, period, dist = parts[1], parts[2], parts[3]
    seed = int(parts[5][1:]) if len(parts) > 5 and parts[5].startswith('s') else None
    return {
        'model': model,
        'index': index,
        'period': period,
        'dist': dist,
        'seed': seed,
        'n': n,
        'exceptions': n_excep,
        'expected': ALPHA * n,
        'var_mean': df['var_forecast'].mean(),
        'var_std': df['var_forecast'].std(),
        'p_uc': kupiec_uc(n_excep, n),
        'p_cc': christoffersen_cc(n_excep, n, df['exceeded']),
        'p_ind': christoffersen_ind(df['exceeded']),
        'p_dq': dq_test(df['return'], df['var_forecast'], df['exceeded']),
        'lopez': losses['lopez'],
        'caporin_reg': losses['caporin_reg'],
        'caporin_firm': losses['caporin_firm'],
    }


def main():
    files = sorted(FORECAST_DIR.glob('*.csv'))
    print(f"Analyzing {len(files)} files...")
    rows = [analyze_file(f) for f in files]
    per_file = pd.DataFrame(rows)
    per_file.to_csv(OUT_DIR / 'analysis_per_file.csv', index=False)
    print(f"Wrote {OUT_DIR / 'analysis_per_file.csv'}")

    # Aggregate GARCHNet across seeds
    gn = per_file[per_file['model'] == 'garchnet'].copy()
    g = per_file[per_file['model'] == 'garch'].copy()

    gn_agg = gn.groupby(['index', 'period', 'dist']).agg(
        gn_excep_mean=('exceptions', 'mean'),
        gn_excep_std=('exceptions', 'std'),
        gn_excep_min=('exceptions', 'min'),
        gn_excep_max=('exceptions', 'max'),
        gn_p_uc_mean=('p_uc', 'mean'),
        gn_p_cc_mean=('p_cc', 'mean'),
        gn_p_dq_mean=('p_dq', 'mean'),
        gn_var_std_mean=('var_std', 'mean'),
    ).reset_index()
    gn_agg.to_csv(OUT_DIR / 'analysis_summary.csv', index=False)
    print(f"Wrote {OUT_DIR / 'analysis_summary.csv'}")

    # Side-by-side comparison
    g_pivot = g[['index', 'period', 'dist', 'exceptions', 'p_uc', 'p_cc', 'p_dq', 'var_std']].rename(
        columns={
            'exceptions': 'g_excep', 'p_uc': 'g_p_uc',
            'p_cc': 'g_p_cc', 'p_dq': 'g_p_dq', 'var_std': 'g_var_std',
        }
    )
    comparison = g_pivot.merge(gn_agg, on=['index', 'period', 'dist'])
    # Reorder columns
    comparison = comparison[['index', 'period', 'dist',
                              'g_excep', 'gn_excep_mean', 'gn_excep_std',
                              'g_p_uc', 'gn_p_uc_mean',
                              'g_p_cc', 'gn_p_cc_mean',
                              'g_p_dq', 'gn_p_dq_mean',
                              'g_var_std', 'gn_var_std_mean']]
    comparison.to_csv(OUT_DIR / 'comparison_table.csv', index=False)
    print(f"Wrote {OUT_DIR / 'comparison_table.csv'}\n")

    # Filter to the periods + indices you have GARCHNet for
    gn_combos = gn[['index', 'period', 'dist']].drop_duplicates()
    g_filtered = g.merge(gn_combos, on=['index', 'period', 'dist'])

    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    with pd.option_context('display.max_columns', None, 'display.width', 200,
                            'display.float_format', lambda v: f'{v:.3f}'):
        print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()