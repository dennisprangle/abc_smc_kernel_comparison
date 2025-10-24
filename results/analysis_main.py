# Load results and output summary tables
# For float format, use .2e for scientific format (3sf) or .3g for decimal format (3sf)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Main results
res = pd.read_csv('results.csv')
eps_table = res.pivot_table(
    index='model',
    columns=['proposal', 'kernel'],
    values='final_epsilon',
    aggfunc=['mean']
)
tex_table = eps_table.to_latex(float_format="%.2e")
print(tex_table)

eps_table = res.pivot_table(
    index='model',
    columns=['proposal', 'kernel'],
    values='final_epsilon',
    aggfunc=['std']
)
tex_table = eps_table.to_latex(float_format="%.2e")
print(tex_table)

loss_table = res.pivot_table(
    index='model',
    columns=['proposal', 'kernel'],
    values='loss',
    aggfunc=['mean']
)
tex_table = loss_table.to_latex(float_format="%.3g")
print(tex_table)

loss_table = res.pivot_table(
    index='model',
    columns=['proposal', 'kernel'],
    values='loss',
    aggfunc=['std']
)
tex_table = loss_table.to_latex(float_format="%.3g")
print(tex_table)

# ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Get unique models
models = res['model'].unique()
res['proposal_kernel'] = res['proposal'] + "_" + res['kernel']

for model in models:
    subset = res[res['model'] == model]
    fit = ols('loss ~ C(proposal_kernel)', data=subset).fit()
    print(f"ANOVA on loss for model: {model}")
    anova_table = sm.stats.anova_lm(fit, typ=2)
    print(anova_table, "\n")
    fit = ols('final_epsilon ~ C(proposal_kernel)', data=subset).fit()
    print(f"ANOVA on epsilon for model: {model}")
    anova_table = sm.stats.anova_lm(fit, typ=2)
    print(anova_table, "\n")
