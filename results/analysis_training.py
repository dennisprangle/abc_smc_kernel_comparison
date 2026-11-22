# Load main results and output tables of the proportion of runtime spent training proposals
# For float format, use .2e for scientific format (3sf) or .3g for decimal format (3sf)

import pandas as pd

res = pd.read_csv('results.csv')
training_table = res.pivot_table(
    index='model',
    columns=['proposal', 'kernel'],
    values='training_proportion',
    aggfunc=['mean']
)
tex_table = training_table.to_latex(float_format="%.3g")
print(tex_table)

training_table = res.pivot_table(
    index='model',
    columns=['proposal', 'kernel'],
    values='training_proportion',
    aggfunc=['std']
)
tex_table = training_table.to_latex(float_format="%.3g")
print(tex_table)

# Plot: training proportion (mean over seeds) by proposal, one panel per model, kernel as colour
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_utils import bottom_legend, TEXT_WIDTH, use_paper_style

use_paper_style()

proposal_order = ['classic_rw', 'classic_ind', 'flow', 'mix']
proposal_names_plot = ['classic RW', 'classic ind', 'flow', 'mix']
kernel_order = ['naive', '1_hit', 'ind_1_hit', 'r_hit_er', 'multi_r_hit_er']
kernel_names_plot = ['ABC-MH', '1-hit', 'ind 1-hit', 'single r-hit ER', 'multi r-hit ER']
model_names_data = ['quadratic', 'MG1_quantile', 'slcp', 'seir', 'gaussian_mixture']
model_names_plot = ['quadratic', 'MG1', 'SLCP', 'SEIR', 'GM']
kernel_colours = dict(zip(kernel_order, sns.color_palette('deep', n_colors=len(kernel_order))))

means = res.groupby(['proposal', 'kernel', 'model'], as_index=False)['training_proportion'].mean()

offset = 0.04  # slight horizontal offset between kernels within a proposal
fig, axes = plt.subplots(1, 5, figsize=(TEXT_WIDTH, 2.7), sharey=True)
for ax, model, model_name in zip(axes, model_names_data, model_names_plot):
    for xi, proposal in enumerate(proposal_order):
        for ki, kernel in enumerate(kernel_order):
            # Combinations that weren't run are skipped, leaving a gap
            y = means[(means['proposal'] == proposal) & (means['kernel'] == kernel)
                      & (means['model'] == model)]['training_proportion']
            x = xi + (ki - (len(kernel_order) - 1) / 2) * offset
            ax.scatter([x] * len(y), y, color=kernel_colours[kernel], s=12, linewidths=0)
    ax.set_title(model_name)
    ax.set_xticks(range(len(proposal_order)), proposal_names_plot, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Training proportion')

# A single legend for all panels
handles = [plt.Line2D([], [], marker='o', linestyle='', color=kernel_colours[k], label=n)
           for k, n in zip(kernel_order, kernel_names_plot)]
bottom_legend(fig, handles, title='Kernel', margin=0.12)
# No creation date, so an unchanged plot gives an identical file
fig.savefig('training_grid_plot.pdf', metadata={'CreationDate': None})
