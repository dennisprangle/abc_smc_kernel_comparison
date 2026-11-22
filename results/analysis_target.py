# Analyse results comparing choice of targets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.ion()

res = pd.read_csv('target.csv')

# Summary tables
eps_table = res.pivot_table(
    index='model',
    columns=['train_target', 'proposal'],
    values='final_epsilon',
    aggfunc=['mean']
)
print(eps_table)

loss_table = res.pivot_table(
    index='model',
    columns=['train_target', 'proposal'],
    values='loss',
    aggfunc=['mean']
)
print(loss_table)

# Plots
from analysis_utils import set_log_ticks, bottom_legend, TEXT_WIDTH, use_paper_style

use_paper_style()

target_order = ['current', 'previous']
target_names_plot = [r'$\mathcal{A}_t$', r'$\mathcal{B}_t$']
target_colours = dict(zip(target_order, sns.color_palette('deep', n_colors=len(target_order))))
target_markers = dict(zip(target_order, ['o', 'X']))
proposal_order = ['classic_rw', 'classic_ind', 'mix', 'flow']
proposal_names_plot = ['classic RW', 'classic ind', 'mix', 'flow']
model_names_data = ['quadratic', 'MG1_quantile', 'slcp', 'seir', 'gaussian_mixture']
model_names_plot = ['quadratic', 'MG1', 'SLCP', 'SEIR', 'GM']

# A fixed proposal order, so x positions don't depend on row order
plot_res = res.copy()
plot_res['proposal'] = pd.Categorical(plot_res['proposal'], categories=proposal_order)

# Fixed seed for seaborn's strip plot jitter, so reruns give the same plot
np.random.seed(1)

fig, axes = plt.subplots(
    2, 5, figsize=(TEXT_WIDTH, 3.6), sharex=True, sharey=False
)

for row, (column, label) in enumerate([('final_epsilon', r'$\epsilon$'), ('loss', 'Loss')]):
    for i, model in enumerate(model_names_data):
        ax = axes[row, i]
        # One strip plot per target so each gets its own marker.
        # Passing the full hue_order dodges each target to its own side of the proposal
        for target in target_order:
            data = plot_res[(plot_res['model'] == model) & (plot_res['train_target'] == target)]
            sns.stripplot(
                data=data, x="proposal", y=column, hue="train_target", hue_order=target_order,
                palette=target_colours, marker=target_markers[target], dodge=True, jitter=0.1,
                alpha=0.8, size=3.5, legend=False, ax=ax
            )
        # seaborn's dodge gap can't be set, so pull points towards their proposal's x position,
        # narrowing both the gap between targets and the jitter
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

axes[1, 0].set_xticks(range(len(proposal_order)), proposal_names_plot)
for ax in axes[1]:
    ax.tick_params(axis='x', labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')

# A single legend for all panels
legend_handles = [
    plt.Line2D([], [], marker=target_markers[target], linestyle='', color=target_colours[target], label=name)
    for target, name in zip(target_order, target_names_plot)
]
bottom_legend(fig, legend_handles, title='Target')
# No creation date, so an unchanged plot gives an identical file
fig.savefig('target_grid_plot.pdf', metadata={'CreationDate': None})

# Paired t-tests (from copilot - check details if we use this in paper)
from scipy.stats import ttest_rel
from analysis_utils import paired_by_seed

long_res = res.melt(
    id_vars=["model", "proposal"],
    value_vars=["final_epsilon"],
    var_name="metric",
    value_name="value",
)

results = []
grouped = res.groupby(['model', 'proposal'])
for (model, proposal), group in grouped:
    current, previous = paired_by_seed(
        group[group['train_target'] == 'current'],
        group[group['train_target'] == 'previous'],
        'final_epsilon'
    )
    
    t_stat, p_value = ttest_rel(current, previous)
    
    # Store the results
    results.append({
        'model': model,
        'proposal': proposal,
        't_stat': t_stat,
        'p_value': p_value
    })

results_df = pd.DataFrame(results)
print(results_df)

long_res = res.melt(
    id_vars=["model", "proposal"],
    value_vars=["loss"],
    var_name="metric",
    value_name="value",
)

results = []
grouped = res.groupby(['model', 'proposal'])
for (model, proposal), group in grouped:
    current, previous = paired_by_seed(
        group[group['train_target'] == 'current'],
        group[group['train_target'] == 'previous'],
        'loss'
    )
    
    t_stat, p_value = ttest_rel(current, previous)
    
    # Store the results
    results.append({
        'model': model,
        'proposal': proposal,
        't_stat': t_stat,
        'p_value': p_value
    })

results_df = pd.DataFrame(results)
print(results_df)