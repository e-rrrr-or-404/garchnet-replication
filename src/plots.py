"""Generate the three figures for the presentation."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_var_overlay(index='spx', period='2020', dist='normal', seed=1):
    g = pd.read_csv(ROOT / f'results/var_forecasts/garch_{index}_{period}_{dist}.csv')
    gn = pd.read_csv(ROOT / f'results/var_forecasts/garchnet_{index}_{period}_{dist}_p20_s{seed}.csv')
    g['date'] = pd.to_datetime(g['date'])
    gn['date'] = pd.to_datetime(gn['date'])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(g['date'], g['return'], 'k.', markersize=3, alpha=0.5, label='Realized return')
    ax.plot(g['date'], g['var_forecast'], 'b-', linewidth=1.5, label='GARCH VaR (2.5%)')
    ax.plot(gn['date'], gn['var_forecast'], 'r-', linewidth=1.5, label=f'GARCHNet VaR (2.5%, seed {seed})')
    # Highlight exceptions
    g_ex = g[g['exceeded'] == 1]
    gn_ex = gn[gn['exceeded'] == 1]
    ax.scatter(g_ex['date'], g_ex['return'], color='blue', s=40, zorder=5, label=f'GARCH excep ({len(g_ex)})')
    ax.scatter(gn_ex['date'], gn_ex['return'], color='red', s=20, zorder=6, marker='x', label=f'GARCHNet excep ({len(gn_ex)})')
    ax.set_title(f'{index.upper()} {period}, {dist} dist: VaR forecasts')
    ax.set_ylabel('Log return')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f'var_overlay_{index}_{period}_{dist}.png', dpi=150)
    plt.close(fig)
    print(f"Saved var_overlay_{index}_{period}_{dist}.png")


def plot_exception_bars():
    df = pd.read_csv(ROOT / 'results/comparison_table.csv')
    df['label'] = df['index'].astype(str) + '\n' + df['period'].astype(str) + '\n' + df['dist'].astype(str)
    x = np.arange(len(df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, df['g_excep'], w, label='GARCH', color='steelblue')
    ax.bar(x + w/2, df['gn_excep_mean'], w, yerr=df['gn_excep_std'],
            label='GARCHNet (mean ± std, 3 seeds)', color='firebrick', capsize=3)
    ax.axhline(6.25, color='black', linestyle='--', linewidth=1, label='Expected (α=2.5%, n=250)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['label'], fontsize=7)
    ax.set_ylabel('Exceptions / 250')
    ax.set_title('Exception counts: GARCH vs GARCHNet across all (index, period, dist)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'exception_bars.png', dpi=150)
    plt.close(fig)
    print("Saved exception_bars.png")


def plot_seed_variance():
    df = pd.read_csv(ROOT / 'results/analysis_per_file.csv')
    gn = df[df['model'] == 'garchnet'].copy()
    gn['combo'] = gn['index'].astype(str) + '_' + gn['period'].astype(str) + '_' + gn['dist'].astype(str)
    combos = sorted(gn['combo'].unique())
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, c in enumerate(combos):
        sub = gn[gn['combo'] == c]
        ax.scatter([i]*len(sub), sub['exceptions'], s=40, alpha=0.7, color='firebrick')
        ax.scatter([i], sub['exceptions'].mean(), s=120, marker='_',
                    color='black', linewidth=2)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combos, rotation=70, fontsize=7, ha='right')
    ax.axhline(6.25, color='black', linestyle='--', linewidth=1, label='Expected')
    ax.set_ylabel('Exceptions / 250')
    ax.set_title('GARCHNet exception counts across 3 seeds (red dots = seeds, black bar = mean)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'seed_variance.png', dpi=150)
    plt.close(fig)
    print("Saved seed_variance.png")


if __name__ == '__main__':
    plot_var_overlay('spx', '2020', 'normal', seed=1)
    plot_var_overlay('spx', '2009', 'normal', seed=1)
    plot_var_overlay('wig20', '2020', 'normal', seed=1)
    plot_exception_bars()
    plot_seed_variance()
    print(f"\nAll figures saved to {FIG_DIR}")