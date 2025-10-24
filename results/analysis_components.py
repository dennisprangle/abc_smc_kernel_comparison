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
res["components"] = res["components"].astype(str) # to handle mixed data types

plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(
    2, 5, figsize=(20, 8), sharex=True, sharey=False
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
        data=data, x="components", y="value", ax=axes[0, i]
    )
    axes[0, i].set_title(model_names_plot[i])
    axes[0, i].set_xlabel('')
    if i == 0:
        axes[0, i].set_ylabel('Epsilon')
    else:
        axes[0, i].set_ylabel('')
    axes[0, i].set_xticklabels([])

long_res = res.melt(
    id_vars=["model", "components"],
    value_vars=["loss"],
    var_name="metric",
    value_name="value",
)

for i, model in enumerate(model_names_data):
    data = long_res[long_res['model'] == model]
    sns.scatterplot(
        data=data, x="components", y="value", ax=axes[1, i]
    )
    axes[1, i].set_title('')
    if i == 0:
        axes[1, i].set_ylabel('Loss')
    else:
        axes[1, i].set_ylabel('')
    axes[1, i].set_xticklabels(['3','4','5','6','7','8','9','10','auto'])

fig.tight_layout()
fig.savefig('components_grid_plot.pdf')

