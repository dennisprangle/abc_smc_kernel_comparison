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

res_main = pd.read_csv('results.csv')
res_main = res_main[res_main['proposal'] == 'mix']

for model in res_main['model'].unique():
    for kernel in res_main['kernel'].unique():
        def_vals = res_def[(res_def['model'] == model) & (res_def['kernel'] == kernel)]['final_epsilon']
        main_vals = res_main[(res_main['model'] == model) & (res_main['kernel'] == kernel)]['final_epsilon']
        _, p_val = ttest_rel(def_vals, main_vals)
        print(f"tests for model={model}, kernel={kernel}")
        print(f"eps: p={p_val:.3g}")
        def_vals = res_def[(res_def['model'] == model) & (res_def['kernel'] == kernel)]['loss']
        main_vals = res_main[(res_main['model'] == model) & (res_main['kernel'] == kernel)]['loss']
        _, p_val = ttest_rel(def_vals, main_vals)
        print(f"loss: p={p_val:.3g}\n")

# Plots
long_res1 = res_def.melt(
    id_vars=["model", "kernel"],
    value_vars=["final_epsilon"],
    var_name="metric",
    value_name="value",
)
long_res1['defensive'] = True

long_res2 = res_main.melt(
    id_vars=["model", "kernel"],
    value_vars=["final_epsilon"],
    var_name="metric",
    value_name="value",
)
long_res2['defensive'] = False

long_res = pd.concat([long_res1, long_res2], ignore_index=True)

model_names_data = ['quadratic', 'MG1_quantile', 'slcp', 'seir', 'gaussian_mixture']
model_names_plot = ['quadratic', 'MG1', 'SLCP', 'SEIR', 'GM']

plt.rcParams.update({'font.size': 10})

fig, axes = plt.subplots(
    2, 5, figsize=(20, 8), sharex=True, sharey=False
)

for i, model in enumerate(model_names_data):
    data = long_res[long_res['model'] == model]
    sns.scatterplot(
        data=data, x="kernel", y="value", hue="defensive", style="defensive", palette="deep", alpha=0.5,  ax=axes[0, i]
    )
    axes[0, i].set_title(model_names_plot[i])
    axes[0, i].set_xlabel('')
    if i == 0:
        axes[0, i].set_ylabel('Epsilon')
    else:
        axes[0, i].set_ylabel('')
    if i < 4:
        axes[0, i].legend_.remove()

long_res1 = res_def.melt(
    id_vars=["model", "kernel"],
    value_vars=["loss"],
    var_name="metric",
    value_name="value",
)
long_res1['defensive'] = True

long_res2 = res_main.melt(
    id_vars=["model", "kernel"],
    value_vars=["loss"],
    var_name="metric",
    value_name="value",
)
long_res2['defensive'] = False

long_res = pd.concat([long_res1, long_res2], ignore_index=True)

for i, model in enumerate(model_names_data):
    data = long_res[long_res['model'] == model]
    sns.scatterplot(
        data=data, x="kernel", y="value", hue="defensive", style="defensive", palette="deep", alpha=0.5,  ax=axes[1, i]
    )
    axes[1, i].set_title('')
    if i == 0:
        axes[1, i].set_ylabel('Loss')
    else:
        axes[1, i].set_ylabel('')
    axes[1, i].legend_.remove()

fig.tight_layout()
fig.savefig('defensive_grid_plot.pdf')