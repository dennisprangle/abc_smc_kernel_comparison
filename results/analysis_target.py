# Analyse results comparing choice of targets
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
plt.rcParams.update({'font.size': 10})

fig, axes = plt.subplots(
    2, 5, figsize=(20, 8), sharex=True, sharey=False
)

# Plot final epsilon
long_res = res.melt(
    id_vars=["model", "proposal", "train_target"],
    value_vars=["final_epsilon"],
    var_name="metric",
    value_name="value",
)

model_names_data = ['quadratic', 'MG1_quantile', 'slcp', 'seir', 'gaussian_mixture']
model_names_plot = ['quadratic', 'MG1', 'SLCP', 'SEIR', 'GM']

for i, model in enumerate(model_names_data):
    data = long_res[long_res['model'] == model]
    sns.scatterplot(
        data=data, x="proposal", y="value", hue="train_target", style="train_target", palette="deep", alpha=0.8,  ax=axes[0, i]
    )
    axes[0, i].set_title(model_names_plot[i])
    axes[0, i].set_xlabel('')
    axes[0, i].set_yscale('log')
    if i == 0:
        axes[0, i].set_ylabel('Epsilon')
    else:
        axes[0, i].set_ylabel('')
    axes[0, i].set_xticklabels([])
    if i < 4:
        axes[0, i].legend_.remove()
    # Single legend in one grid cell for whole plot
    handles, labels = axes[0, 4].get_legend_handles_labels()
    # print(labels) # Shows ['current', 'previous'], so need to reverse order of handles
    custom_labels = [r'$\mathcal{A}_t$', r'$\mathcal{B}_t$']
    axes[0, 4].legend(handles=handles[::-1], labels=custom_labels, title='Target')

# Plot loss
long_res = res.melt(
    id_vars=["model", "proposal", "train_target"],
    value_vars=["loss"],
    var_name="metric",
    value_name="value",
)

for i, model in enumerate(model_names_data):
    data = long_res[long_res['model'] == model]
    sns.scatterplot(
        data=data, x="proposal", y="value", hue="train_target", style="train_target", palette="deep", alpha=0.8,  ax=axes[1, i]
    )
    axes[1, i].set_title('')
    if i == 0:
        axes[1, i].set_ylabel('Loss')
    else:
        axes[1, i].set_ylabel('')
    axes[1, i].set_xticklabels(['classic RW', 'classic ind', 'mix', 'flow'])
    axes[1, i].legend_.remove()

fig.tight_layout()
fig.savefig('target_grid_plot.pdf')

# Paired t-tests (from copilot - check details if we use this in paper)
from scipy.stats import ttest_rel

long_res = res.melt(
    id_vars=["model", "proposal"],
    value_vars=["final_epsilon"],
    var_name="metric",
    value_name="value",
)

results = []
grouped = res.groupby(['model', 'proposal'])
for (model, proposal), group in grouped:
    current = group[group['train_target'] == 'current']['final_epsilon']
    previous = group[group['train_target'] == 'previous']['final_epsilon']
    
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
    current = group[group['train_target'] == 'current']['loss']
    previous = group[group['train_target'] == 'previous']['loss']
    
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