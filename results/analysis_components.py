# Analyse results on number of components

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

res = pd.read_csv('components.csv')
res['components'] = res.apply(
    lambda row: row['min_mixture_components'] if row['min_mixture_components'] == row['max_mixture_components'] else 'auto',
    axis=1
)
eps_table = res.pivot_table(
    index='model',
    columns=['components'],
    values='final_epsilon',
    aggfunc=['mean']
)
tex_table = eps_table.to_latex(float_format="%.3e")
print(tex_table)
loss_table = res.pivot_table(
    index='model',
    columns=['components'],
    values='loss',
    aggfunc=['mean']
)
tex_table = loss_table.to_latex(float_format="%.3e")
print(tex_table)

# Plots
# Strings to handle mixed data types, and a fixed category order so x positions don't depend on row order
components_order = [str(c) for c in range(3, 11)] + ['auto']
res["components"] = pd.Categorical(res["components"].astype(str), categories=components_order)

from analysis_utils import TEXT_WIDTH, use_paper_style
use_paper_style()
fig, axes = plt.subplots(
    2, 5, figsize=(TEXT_WIDTH, 3.6), sharex=True, sharey=False
)
model_names_data = ['quadratic', 'MG1_quantile', 'slcp', 'seir', 'gaussian_mixture']
model_names_plot = ['quadratic', 'MG1', 'SLCP', 'SEIR', 'GM']

long_res = res.melt(
    id_vars=["model", "components"],
    value_vars=["final_epsilon"],
    var_name="metric",
    value_name="value",
)

for i, model in enumerate(model_names_data):
    data = long_res[long_res['model'] == model]
    sns.scatterplot(
        data=data, x="components", y="value", ax=axes[0, i], s=12, linewidth=0
    )
    # A fixed title gap leaves room for the y-axis offset text, keeping titles level
    axes[0, i].set_title(model_names_plot[i], pad=10)
    if model == 'gaussian_mixture':
        # Scientific notation, with offset text like the quadratic panel
        axes[0, i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[0, i].set_xlabel('')
    if i == 0:
        axes[0, i].set_ylabel(r'$\epsilon$')
    else:
        axes[0, i].set_ylabel('')
    axes[0, i].tick_params(axis='x', labelbottom=False)

long_res = res.melt(
    id_vars=["model", "components"],
    value_vars=["loss"],
    var_name="metric",
    value_name="value",
)

for i, model in enumerate(model_names_data):
    data = long_res[long_res['model'] == model]
    sns.scatterplot(
        data=data, x="components", y="value", ax=axes[1, i], s=12, linewidth=0
    )
    axes[1, i].set_title('')
    axes[1, i].tick_params(axis='x', labelrotation=45)
    plt.setp(axes[1, i].get_xticklabels(), ha='right', rotation_mode='anchor')

    for ax in axes[:, i]:
        ax.grid(alpha=0.3)

    if i == 0:
        axes[1, i].set_ylabel('Loss')
    else:
        axes[1, i].set_ylabel('')

fig.tight_layout()
fig.savefig('components_grid_plot.pdf', metadata={'CreationDate': None})

# ANOVA, excluding automatic selection of the number of components
# Back to plain strings, as the unused 'auto' category would add an all-zero column to the design matrix
anova_res = res[res["components"] != "auto"].copy()
anova_res["components"] = anova_res["components"].astype(str)
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Get unique models
models = anova_res['model'].unique()

for model in models:
    subset = anova_res[anova_res['model'] == model]
    fit = ols('loss ~ C(components)', data=subset).fit()
    print(f"ANOVA on loss for model: {model}")
    anova_table = sm.stats.anova_lm(fit, typ=2)
    print(anova_table, "\n")
    fit = ols('final_epsilon ~ C(components)', data=subset).fit()
    print(f"ANOVA on epsilon for model: {model}")
    anova_table = sm.stats.anova_lm(fit, typ=2)
    print(anova_table, "\n")
