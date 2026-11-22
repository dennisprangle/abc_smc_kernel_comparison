# Analyse results on number of hits for each r-hit kernel variant

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from analysis_utils import set_log_ticks, bottom_legend, TEXT_WIDTH, use_paper_style

kernels = ['r_hit', 'r_hit_ea', 'r_hit_er', 'multi_r_hit', 'multi_r_hit_ea', 'multi_r_hit_er']

all_res = pd.read_csv("hits.csv")

# Plots: mean over seeds against number of hits, one line per kernel.
# Single-proposal r_hit and r_hit_ea are omitted, as most of their runs have infinite final_epsilon
plot_kernels = {
    'r_hit_er': 'single, early reject',
    'multi_r_hit': 'multi',
    'multi_r_hit_ea': 'multi, early accept',
    'multi_r_hit_er': 'multi, early reject',
}
kernel_colours = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']

use_paper_style()
fig, axes = plt.subplots(
    2, 5, figsize=(TEXT_WIDTH, 3.6), sharex=True, sharey=False
)
model_names_data = ['quadratic', 'MG1_quantile', 'slcp', 'seir', 'gaussian_mixture']
model_names_plot = ['quadratic', 'MG1', 'SLCP', 'SEIR', 'GM']
means = all_res.groupby(['model', 'kernel', 'hits'])[['final_epsilon', 'loss']].mean()

for row, (column, label) in enumerate([('final_epsilon', r'$\epsilon$'), ('loss', 'Loss')]):
    for i, model in enumerate(model_names_data):
        ax = axes[row, i]
        for kernel, colour in zip(plot_kernels, kernel_colours):
            values = means.loc[(model, kernel), column]
            ax.plot(
                values.index, values.to_numpy(), marker='o', markersize=3,
                color=colour, linewidth=1, label=plot_kernels[kernel]
            )
        if column == 'final_epsilon':
            ax.set_yscale('log')
            set_log_ticks(ax)
            ax.set_title(model_names_plot[i])
        ax.set_ylabel(label if i == 0 else '')
        ax.set_xlabel('Hits' if row == 1 else '')
        ax.set_xticks(sorted(all_res['hits'].unique()))
        ax.grid(alpha=0.3)

# A single legend for all panels
handles, labels = axes[0, 0].get_legend_handles_labels()
bottom_legend(fig, handles, labels)
fig.savefig('hits_grid_plot.pdf', metadata={'CreationDate': None})

# Number of runs (out of 5 seeds) with infinite final_epsilon,
# i.e. which never finished an iteration after the first
all_res['infinite_epsilon'] = ~np.isfinite(all_res['final_epsilon'])
inf_table = all_res.pivot_table(
    index=['model', 'hits'],
    columns='kernel',
    values='infinite_epsilon',
    aggfunc='sum'
)[kernels]
print("Number of runs (out of 5) with infinite final epsilon\n")
print(inf_table.to_string(), "\n")

# ANOVAs for the effect of number of hits, one row per kernel and model.
# Epsilon ANOVAs omit runs with infinite final_epsilon, and are nan when there aren't enough runs left
def anova_hits(data, column):
    """F statistic and p-value from a one-way ANOVA of `column` on number of hits"""
    if data['hits'].nunique() < 2 or len(data) <= data['hits'].nunique():
        return np.nan, np.nan
    fit = ols(f'{column} ~ C(hits)', data=data).fit()
    anova_table = sm.stats.anova_lm(fit, typ=2)
    return anova_table.loc['C(hits)', 'F'], anova_table.loc['C(hits)', 'PR(>F)']

anova_rows = []
for kernel in kernels:
    for model in model_names_data:
        subset = all_res[(all_res['kernel'] == kernel) & (all_res['model'] == model)]
        finite = subset[np.isfinite(subset['final_epsilon'])]
        loss_F, loss_p = anova_hits(subset, 'loss')
        eps_F, eps_p = anova_hits(finite, 'final_epsilon')
        anova_rows.append({
            'kernel': kernel, 'model': model,
            ('loss', 'F'): loss_F, ('loss', 'p'): loss_p,
            ('epsilon', 'F'): eps_F, ('epsilon', 'p'): eps_p,
        })
anova_table = pd.DataFrame(anova_rows).set_index(['kernel', 'model'])
anova_table.columns = pd.MultiIndex.from_tuples(anova_table.columns)
print("ANOVAs for the effect of number of hits (epsilon omits runs with infinite final epsilon)\n")
print(anova_table.to_string(float_format=lambda x: f"{x:.3g}"))
