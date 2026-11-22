# Analyse results of flow architecture experiments
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Flow architecture results
res = pd.read_csv("flow.csv")

eps_table = res.pivot_table(
    index='model',
    columns=['flow'],
    values='final_epsilon',
    aggfunc=['mean', 'std']
)
# For float format, use .2e for scientific format (3sf) or .3g for decimal format (3sf)
tex_table = eps_table.to_latex(float_format="%.2e")
print(tex_table)

loss_table = res.pivot_table(
    index='model',
    columns=['flow'],
    values='loss',
    aggfunc=['mean', 'std']
)
tex_table = loss_table.to_latex(float_format="%.3g")
print(tex_table)

# Swarm plots
from analysis_utils import bottom_legend, TEXT_WIDTH, use_paper_style
use_paper_style()

flow_order = ['spline', 'spline_two_layers', 'coupling']
flow_names_plot = ['spline', 'spline (two layers)', 'coupling']
flow_colours = dict(zip(flow_order, sns.color_palette(n_colors=len(flow_order))))
model_names_data = ['quadratic', 'MG1_quantile', 'slcp', 'seir', 'gaussian_mixture']
model_names_plot = ['quadratic', 'MG1', 'SLCP', 'SEIR', 'GM']

fig, axes = plt.subplots(
    2, len(model_names_data), figsize=(TEXT_WIDTH, 3.6), sharex=True, sharey=False
)

for i, model in enumerate(model_names_data):
    data = res[res['model'] == model]
    # Epsilon plot (top row)
    sns.swarmplot(
        data=data, x='flow', y='final_epsilon', order=flow_order, hue='flow', palette=flow_colours, legend=False, size=3.5, ax=axes[0, i]
    )
    axes[0, i].set_title(model_names_plot[i])
    axes[0, i].set_xlabel('')
    if i == 0:
        axes[0, i].set_ylabel(r'$\epsilon$')
    else:
        axes[0, i].set_ylabel('')
    axes[0, i].tick_params(axis='x', bottom=False)


    # Loss plot (bottom row)
    sns.swarmplot(
        data=data, x='flow', y='loss', order=flow_order, hue='flow', palette=flow_colours, legend=False, size=3.5, ax=axes[1, i]
    )
    axes[1, i].set_xlabel('')
    axes[1, i].tick_params(axis='x', bottom=False, labelbottom=False)
    if i == 0:
        axes[1, i].set_ylabel('Loss')
    else:
        axes[1, i].set_ylabel('')

    axes[1, i].set_title('')

    for ax in axes[:, i]:
        ax.grid(alpha=0.3)

# Flows are identified by a single legend rather than x-axis labels
legend_handles = [
    plt.Line2D([], [], marker='o', linestyle='', color=flow_colours[flow], label=name)
    for flow, name in zip(flow_order, flow_names_plot)
]
bottom_legend(fig, legend_handles)
fig.savefig('flow_grid_plot.pdf', metadata={'CreationDate': None})

# ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Get unique models
models = res['model'].unique()

for model in models:
    subset = res[res['model'] == model]
    fit = ols('loss ~ C(flow)', data=subset).fit()
    print(f"ANOVA on loss for model: {model}")
    anova_table = sm.stats.anova_lm(fit, typ=2)
    print(anova_table, "\n")
    fit = ols('final_epsilon ~ C(flow)', data=subset).fit()
    print(f"ANOVA on epsilon for model: {model}")
    anova_table = sm.stats.anova_lm(fit, typ=2)
    print(anova_table, "\n")
