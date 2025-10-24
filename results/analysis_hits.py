# Analyse results on number of hits for multi-r-hit

import pandas as pd

# Multiple proposal results on number of hits
res = pd.read_csv("hits.csv")
res.sort_values(["model", "hits"])

eps_table = res.pivot_table(
    index='model',
    columns=['hits'],
    values='final_epsilon',
    aggfunc=['mean', 'std']
)
# For float format, use .2e for scientific format (3sf) or .3g for decimal format (3sf)
tex_table = eps_table.to_latex(float_format="%.2e")
print(tex_table)

loss_table = res.pivot_table(
    index='model',
    columns=['hits'],
    values='loss',
    aggfunc=['mean', 'std']
)
tex_table = loss_table.to_latex(float_format="%.3g")
print(tex_table)

# ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Get unique models
models = res['model'].unique()

for model in models:
    subset = res[res['model'] == model]
    fit = ols('loss ~ C(hits)', data=subset).fit()
    print(f"ANOVA on loss for model: {model}")
    anova_table = sm.stats.anova_lm(fit, typ=2)
    print(anova_table, "\n")
    fit = ols('final_epsilon ~ C(hits)', data=subset).fit()
    print(f"ANOVA on epsilon for model: {model}")
    anova_table = sm.stats.anova_lm(fit, typ=2)
    print(anova_table, "\n")
