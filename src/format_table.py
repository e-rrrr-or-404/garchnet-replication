"""Format the comparison table for the presentation."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT / 'results/comparison_table.csv')

df['gn_excep'] = df.apply(
    lambda r: f"{r['gn_excep_mean']:.1f} ± {r['gn_excep_std']:.1f}",
    axis=1,
)

show = df[['index', 'period', 'dist', 'g_excep', 'gn_excep',
           'g_p_uc', 'gn_p_uc_mean', 'g_p_dq', 'gn_p_dq_mean']]
show.columns = ['Index', 'Period', 'Dist', 'GARCH_excep',
                'GARCHNet_excep (mean±std)', 'GARCH_p_UC', 'GN_p_UC',
                'GARCH_p_DQ', 'GN_p_DQ']
show.to_csv(ROOT / 'results/presentation_table.csv', index=False)
print(show.to_string(index=False))