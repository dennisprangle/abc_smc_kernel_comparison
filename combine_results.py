"""Combine the summary files saved by parallel runs (using --results_dir) into one csv

Usage: python combine_results.py <results_dir> <output_csv>
"""
import sys
from pathlib import Path
import pandas as pd

results_dir, output_path = sys.argv[1:]
files = sorted(Path(results_dir).glob('*.csv'))
if len(files) == 0:
    raise FileNotFoundError(f'No csv files in {results_dir}')
res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Sort so that runs with the same settings are next to each other, in seed order
result_columns = ['final_epsilon', 'epsilon_at_half_time', 'loss', 'training_proportion']
setting_columns = [c for c in res.columns if c not in result_columns + ['seed']]
res = res.sort_values(setting_columns + ['seed'])

res.to_csv(output_path, index=False)
print(f'Combined {len(res)} runs into {output_path}')
