# Load results and output summary tables
# For float format, use .2e for scientific format (3sf) or .3g for decimal format (3sf)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Defensive results
res_def = pd.read_csv('defensive.csv')

eps_table = res_def.pivot_table(
    index='model',
    columns=['kernel'],
    values='final_epsilon',
    aggfunc=['mean']
)
tex_table = eps_table.to_latex(float_format="%.2e")
print(tex_table)

loss_table = res_def.pivot_table(
    index='model',
    columns=['kernel'],
    values='loss',
    aggfunc=['mean']
)
tex_table = loss_table.to_latex(float_format="%.3g")
print(tex_table)

#Tests
from scipy.stats import ttest_rel
from analysis_utils import paired_by_seed

res_main = pd.read_csv('results.csv')
res_main = res_main[res_main['proposal'] == 'mix']

for model in res_main['model'].unique():
    for kernel in res_main['kernel'].unique():
        def_subset = res_def[(res_def['model'] == model) & (res_def['kernel'] == kernel)]
        main_subset = res_main[(res_main['model'] == model) & (res_main['kernel'] == kernel)]
        def_vals, main_vals = paired_by_seed(def_subset, main_subset, 'final_epsilon')
        _, p_val = ttest_rel(def_vals, main_vals)
        print(f"tests for model={model}, kernel={kernel}")
        print(f"eps: p={p_val:.3g}")
        def_vals, main_vals = paired_by_seed(def_subset, main_subset, 'loss')
        _, p_val = ttest_rel(def_vals, main_vals)
        print(f"loss: p={p_val:.3g}\n")

# Plots
import numpy as np
from analysis_utils import set_log_ticks, bottom_legend, TEXT_WIDTH, use_paper_style

use_paper_style()

defensive_order = [False, True]
defensive_names_plot = ['none', '0.1']
defensive_colours = dict(zip(defensive_order, sns.color_palette('deep', n_colors=len(defensive_order))))
defensive_markers = dict(zip(defensive_order, ['o', 'X']))
kernel_order = ['naive', '1_hit', 'ind_1_hit', 'r_hit_er', 'multi_r_hit_er']
kernel_names_plot = ['ABC-MH', '1-hit', 'ind 1-hit', 'single r-hit ER', 'multi r-hit ER']
model_names_data = ['quadratic', 'MG1_quantile', 'slcp', 'seir', 'gaussian_mixture']
model_names_plot = ['quadratic', 'MG1', 'SLCP', 'SEIR', 'GM']

plot_res = pd.concat(
    [res_def.assign(defensive=True), res_main.assign(defensive=False)], ignore_index=True
)
# A fixed kernel order, so x positions don't depend on row order
plot_res['kernel'] = pd.Categorical(plot_res['kernel'], categories=kernel_order)

# Fixed seed for seaborn's strip plot jitter, so reruns give the same plot
np.random.seed(1)

fig, axes = plt.subplots(
    2, 5, figsize=(TEXT_WIDTH, 4.0), sharex=True, sharey=False
)

for row, (column, label) in enumerate([('final_epsilon', r'$\epsilon$'), ('loss', 'Loss')]):
    for i, model in enumerate(model_names_data):
        ax = axes[row, i]
        # One strip plot per setting so each gets its own marker.
        # Passing the full hue_order dodges each setting to its own side of the kernel
        for defensive in defensive_order:
            data = plot_res[(plot_res['model'] == model) & (plot_res['defensive'] == defensive)]
            sns.stripplot(
                data=data, x="kernel", y=column, hue="defensive", hue_order=defensive_order,
                palette=defensive_colours, marker=defensive_markers[defensive], dodge=True, jitter=0.1,
                alpha=0.8, size=3.5, legend=False, ax=ax
            )
        # seaborn's dodge gap can't be set, so pull points towards their kernel's x position,
        # narrowing both the gap between settings and the jitter
        for points in ax.collections:
            x, y = points.get_offsets().T
            centre = np.round(x)
            points.set_offsets(np.column_stack([centre + 0.5 * (x - centre), y]))
        ax.set_xlabel('')
        ax.set_ylabel(label if i == 0 else '')
        if column == 'final_epsilon':
            ax.set_yscale('log')
            set_log_ticks(ax)
            ax.set_title(model_names_plot[i])
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.grid(alpha=0.3)

axes[1, 0].set_xticks(range(len(kernel_order)), kernel_names_plot)
for ax in axes[1]:
    ax.tick_params(axis='x', labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')

# A single legend for all panels
legend_handles = [
    plt.Line2D([], [], marker=defensive_markers[d], linestyle='', color=defensive_colours[d], label=name)
    for d, name in zip(defensive_order, defensive_names_plot)
]
bottom_legend(fig, legend_handles, title='Defensive proportion')
# No creation date, so an unchanged plot gives an identical file
fig.savefig('defensive_grid_plot.pdf', metadata={'CreationDate': None})
