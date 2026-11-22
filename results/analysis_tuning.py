# Load MG1 tuning results (number of particles and epsilon reduction parameter) and output summary tables and a plot
# For float format, use .2e for scientific format (3sf) or .3g for decimal format (3sf)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

res = pd.read_csv('tuning_mg1.csv')

for column, float_format in [('final_epsilon', '%.3g'), ('loss', '%.3g')]:
    table = res.pivot_table(
        index=['n_particles', 'erp'],
        columns=['proposal', 'kernel'],
        values=column,
        aggfunc=['mean']
    )
    print(table.to_latex(float_format=float_format))

# Plot
from analysis_utils import set_log_ticks, bottom_legend, TEXT_WIDTH, use_paper_style

use_paper_style()

proposal_order = ['classic_rw', 'mix', 'flow']
proposal_names_plot = ['classic RW', 'mix', 'flow']
kernel_order = ['naive', '1_hit', 'ind_1_hit']
kernel_names_plot = ['ABC-MH', '1-hit', 'ind 1-hit']
n_particles_order = [500, 1000, 2000]
n_particles_colours = dict(zip(n_particles_order, sns.color_palette('deep', n_colors=len(n_particles_order))))
erp_order = [0.25, 0.5, 0.75]
erp_markers = dict(zip(erp_order, ['o', 's', '^']))

# Mean final epsilon over seeds for each setting
means = res.groupby(['proposal', 'kernel', 'n_particles', 'erp'])['final_epsilon'].mean()

# Within each panel, one x position per (n_particles, erp) setting,
# grouped by n_particles, with a gap between groups
settings = [(n, e) for n in n_particles_order for e in erp_order]
x_positions = {(n, e): i * (len(erp_order) + 1) + j for i, n in enumerate(n_particles_order) for j, e in enumerate(erp_order)}

# One panel per proposal (row) and kernel (column)
fig, axes = plt.subplots(
    len(proposal_order), len(kernel_order), figsize=(TEXT_WIDTH, 4.8), sharex=True, sharey=True
)
for row, (proposal, proposal_name) in enumerate(zip(proposal_order, proposal_names_plot)):
    for col, (kernel, kernel_name) in enumerate(zip(kernel_order, kernel_names_plot)):
        ax = axes[row, col]
        if (proposal, kernel) not in means.index.droplevel(['n_particles', 'erp']):
            # Combination not run (classic RW can't be used with the independence kernel)
            ax.axis('off')
            continue
        for n_particles, erp in settings:
            ax.plot(
                x_positions[(n_particles, erp)], means[(proposal, kernel, n_particles, erp)],
                linestyle='', marker=erp_markers[erp], color=n_particles_colours[n_particles], markersize=4
            )
        ax.set_title(f'{proposal_name} {kernel_name}')
        ax.set_ylabel(r'$\epsilon$' if col == 0 else '')
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.grid(axis='y', alpha=0.3)

# Shared log y-axis: set the ticks once all panels are drawn
axes[0, 0].set_yscale('log')
set_log_ticks(axes[0, 0])

# A single legend: colours for number of particles, then markers for erp
legend_handles = [
    plt.Line2D([], [], marker='o', linestyle='', color=n_particles_colours[n], label=f'N = {n}')
    for n in n_particles_order
] + [
    plt.Line2D([], [], marker=erp_markers[e], linestyle='', color='grey', label=f'erp = {e}')
    for e in erp_order
]
bottom_legend(fig, legend_handles, margin=0.1)
# No creation date, so an unchanged plot gives an identical file
fig.savefig('tuning_mg1_grid_plot.pdf', metadata={'CreationDate': None})
