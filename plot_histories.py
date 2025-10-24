import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

args_parser = argparse.ArgumentParser(description='List options for comparison plots')
args_parser.add_argument('filenames', nargs='+', help='CSV files to plot')
args_parser.add_argument('--legend', nargs='+', help='Legend labels for each file')
args = args_parser.parse_args()

histories = {}
for fname in args.filenames:
    df = pd.read_csv(fname)
    name = fname.split('.')[0] # remove extension
    histories[name] = df

if args.legend is None:
    args.legend = histories.keys()

plt.ion()

#plot epsilon history over time
plt.figure()
for name, hist in histories.items():
    plt.plot(np.array(hist['time']), np.array(hist['epsilon']), '-o', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('$\epsilon$', fontsize=15)
plt.yscale('log')
plt.legend(args.legend)

#plot epsilon history over iterations
plt.figure()
for name, hist in histories.items():
    plt.plot(np.array(hist['iteration']), np.array(hist['epsilon']), '-o', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('$\epsilon$', fontsize=15)
plt.yscale('log')
plt.legend(args.legend)

#plot MCMC acceptances over time
plt.figure()
for name, hist in histories.items():
    plt.plot(np.array(hist['time']), np.array(hist['mcmc_acceptances']), '-o', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('MCMC acceptances', fontsize=15)
plt.legend(args.legend)

input('Press enter to exit')