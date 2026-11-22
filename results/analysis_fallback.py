# Analysis investigating fallback
import pandas as pd

# Fallback results
res = pd.read_csv('fallback.csv')
fallback_table = res.pivot_table(
    index='model',
    columns=['proposal', 'fallback'],
    values='final_epsilon',
    aggfunc=['mean']
)
tex_table = fallback_table.to_latex(float_format="%.3e")
print(tex_table)

fallback_table = res.pivot_table(
    index='model',
    columns=['proposal', 'fallback'],
    values='loss',
    aggfunc=['mean']
)
tex_table = fallback_table.to_latex(float_format="%.3e")
print(tex_table)

## t-test whether fallback has an effect
from scipy.stats import ttest_rel
from analysis_utils import paired_by_seed
for prop in ['mix', 'flow']:
    for model in res['model'].unique().tolist():
        res1 = res.loc[(res['proposal'] == prop) & 
                       (res['fallback'] == True) & 
                       (res['model'] == model)]
        res2 = res.loc[(res['proposal'] == prop) & 
                       (res['fallback'] == False) & 
                       (res['model'] == model)]
        loss1, loss2 = paired_by_seed(res1, res2, 'loss')
        t_stat, p_value = ttest_rel(loss1, loss2)
        print(f"Model: {model}, Proposal: {prop}, p-value (loss): {p_value}")
        print(f"Differences: ", loss1-loss2) # Shows what's going on for nan p-values (in these cases loss1 and loss2 are identical)

for prop in ['mix', 'flow']:
    for model in res['model'].unique().tolist():
        res1 = res.loc[(res['proposal'] == prop) & 
                       (res['fallback'] == True) & 
                       (res['model'] == model)]
        res2 = res.loc[(res['proposal'] == prop) & 
                       (res['fallback'] == False) & 
                       (res['model'] == model)]
        loss1, loss2 = paired_by_seed(res1, res2, 'final_epsilon')
        t_stat, p_value = ttest_rel(loss1, loss2)
        print(f"Model: {model}, Proposal: {prop}, p-value (eps): {p_value}")

