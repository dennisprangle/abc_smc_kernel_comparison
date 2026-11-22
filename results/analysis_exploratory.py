import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import argparse
from analysis_utils import bottom_legend, TEXT_WIDTH, use_paper_style

use_paper_style()

folder_path = "histories_quadratic/"

args_parser = argparse.ArgumentParser(description='List options for comparison plots')
args_parser.add_argument('filenames', nargs='+', help='CSV files to plot')
args_parser.add_argument('--legend', nargs='+', help='Legend labels for each file')
args = args_parser.parse_args(['classic_ind_ind_1_hit_history.csv',
                               'mix_ind_1_hit_history.csv', 
                               'flow_ind_1_hit_history.csv',
                               'classic_rw_1_hit_history.csv',
                               'classic_ind_1_hit_history.csv', 
                               'mix_1_hit_history.csv', 
                               'flow_1_hit_history.csv',
                               'classic_rw_naive_history.csv',
                               'classic_ind_naive_history.csv',
                               'mix_naive_history.csv', 
                               'flow_naive_history.csv',
                               'classic_rw_r_hit_er_history.csv',
                               'classic_ind_r_hit_er_history.csv',
                               'mix_r_hit_er_history.csv',
                               'flow_r_hit_er_history.csv',
                               'classic_rw_multi_r_hit_er_history.csv',
                               'classic_ind_multi_r_hit_er_history.csv',
                               'mix_multi_r_hit_er_history.csv',
                               'flow_multi_r_hit_er_history.csv',])

names = {
    'naive': 'ABC-MH',
    '1_hit': 'One-hit',
    'ind_1_hit': 'Independence\none-hit',
    'r_hit_er': 'Single proposal r-hit\n(early rejection)',
    'multi_r_hit_er': 'Multiple proposal r-hit\n(early rejection)'
}

histories = {}
cols = {'classicind': '#E69F00', 'classicrw': '#0072B2', 'flow': '#009E73', 'mix': '#CC79A7'}  # Okabe-Ito colourblind-safe palette

#TODO: remove?
mark = {'classicind': '-o','classicrw': '-o', 'flow': '-o', 'mix': '-o'}

for fname in args.filenames:
    df = pd.read_csv(folder_path+fname)
    name = fname.split('_history')[0] # remove "_history.csv"
    name = name.replace('classic_ind', 'classicind').replace('classic_rw', 'classicrw')
    histories[name]=df



plt.ion()

# Legend below the plot, with no box, as in the analysis scripts
def add_bottom_legend(fig, handles_or_labels, labels=None, **kwargs):
    args = (handles_or_labels,) if labels is None else (handles_or_labels, labels)
    fig.legend(*args, loc='lower center', ncol=4, frameon=False, **kwargs)

# Each row of a grid is one plot type, with one panel per proposal kernel
# and the y-axis shared along the row
time_xlim = (0., 3600.)
eps_range = (1e-6, 1.1)
rows = {
    'eps_time': dict(x='time', y='epsilon', xlabel='Time (s)', ylabel='$\\epsilon$', yscale='log', xlim=time_xlim, ylim=(1e-6, 1.5)),
    'eps_it': dict(x='iteration', y='epsilon', xlabel='Iterations', ylabel='$\\epsilon$', yscale='log', ylim=(1e-6, 1.5)),
    'loss_time': dict(x='time', y='wloss', xlabel='Time (s)', ylabel='Wasserstein loss', yscale='log', xlim=time_xlim),
    'loss_it': dict(x='iteration', y='wloss', xlabel='Iterations', ylabel='Wasserstein loss', yscale='log'),
    'loss_eps': dict(x='epsilon', y='wloss', xlabel='$\\epsilon$', ylabel='Wasserstein loss', yscale='log', xlim=eps_range, ylim=(0.015, 1)),
    'acc_time': dict(x='time', y='mcmc_acceptances', xlabel='Time (s)', ylabel='MCMC acceptance rate', xlim=time_xlim, ylim=(0., 1.), ydiv=1000),
    'acc_eps': dict(x='epsilon', y='mcmc_acceptances', xlabel='$\\epsilon$', ylabel='MCMC acceptance rate', xlim=eps_range, ylim=(0., 1.), ydiv=1000),
    'time_it': dict(x='iteration', y='time', xlabel='Iterations', ylabel='Time (s)', yscale='log', ylim=(1e0, 1e4)),
}

# One figure per group of rows: columns are proposal kernels
def plot_grid(row_names, filename, proposals, data=histories, titles=names, colors=cols, markers=mark, save=False):
    fig, axes = plt.subplots(len(row_names), len(proposals), figsize=(TEXT_WIDTH, 1.7*len(row_names) + 0.5),
                             sharey='row', squeeze=False)
    for i, row_name in enumerate(row_names):
        row = rows[row_name]
        for j, proposal in enumerate(proposals):
            ax = axes[i, j]
            for name, hist in data.items():
                if proposal == "_".join(name.split("_")[1:]):
                    kernel = name.split('_')[0]
                    ax.plot(np.array(hist[row['x']]), np.array(hist[row['y']])/row.get('ydiv', 1), markers[kernel], linewidth=1, markersize=2, alpha=0.7, color=colors[kernel])
            ax.set(xlabel=row['xlabel'], yscale=row.get('yscale', 'linear'))
            if 'xlim' in row:
                ax.set_xlim(row['xlim'])
            if 'ylim' in row:
                ax.set_ylim(row['ylim'])
            if row['x'] == 'iteration':
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(titles[proposal])
            if j == 0:
                ax.set_ylabel(row['ylabel'])
    # A single legend for all panels
    handles = [plt.Line2D([0], [0], marker='o', color=colors[k], lw=1, markersize=3) for k in colors]
    bottom_legend(fig, handles, list(colors), margin=0.06)
    if save == True:
        fig.savefig(filename, metadata={'CreationDate': None})
    plt.close(fig)


#TODO: remove or update colors       
#boxplot time distribution
def plot_time_boxplots(data = histories, save = False):
    fig, ax = plt.subplots(figsize=(6, 4))  # Crea la figura

    colors = [] 
    labels = []  
    positions = []  

    for i, (label, data) in enumerate(histories.items()):
        if 'classic' in label:
            color = 'dodgerblue'
            labels.append('')
        elif 'flow' in label:
            color = 'green'
            labels.append('')
        elif 'mix' in label:
            color = 'orange'
            labels.append("_".join(label.split("_")[1:]))
        else:
            color = 'gray'  
            labels.append('')

        # add label and color for each boxplot
        colors.append(color)
        positions.append(i)

        #boxplot
        ax.boxplot(np.diff(data['time']), positions=[i], patch_artist=True, 
                   boxprops=dict(facecolor=color, color=color),
                   medianprops=dict(color='black'),
                   showfliers=False)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=15)
    
    for i in range(3, len(positions), 3):
     ax.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)


    # legend
    custom_lines = [plt.Line2D([0], [0], color='dodgerblue', lw=4),
                    plt.Line2D([0], [0], color='green', lw=4),
                    plt.Line2D([0], [0], color='orange', lw=4)]
    ax.grid(alpha=0.3, axis='y')
    add_bottom_legend(fig, custom_lines, ['Classic', 'Flow', 'Mix'], fontsize=15)
    ax.set_xlabel('Proposal', fontsize=16)
    ax.set_ylabel('Time(s)', fontsize=16)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if save == True:
        plt.savefig('boxplot_time.pdf')
    plt.show()
    
#boxplot nr of acceptances distribution
def plot_acceptances_boxplots(data = histories, save = False):
    fig, ax = plt.subplots(figsize=(6, 4))  # Crea la figura

    colors = [] 
    labels = []  
    positions = []  

    for i, (label, data) in enumerate(histories.items()):
        if 'classic' in label:
            color = 'dodgerblue'
            labels.append('')
        elif 'flow' in label:
            color = 'green'
            labels.append('')
        elif 'mix' in label:
            color = 'orange'
            labels.append("_".join(label.split("_")[1:]))
        else:
            color = 'gray'  
            labels.append('')

        # add label and color for each boxplot
        colors.append(color)
        positions.append(i)

        #boxplot
        ax.boxplot(data['mcmc_acceptances'], positions=[i], patch_artist=True, 
                   boxprops=dict(facecolor=color, color=color),
                   medianprops=dict(color='black'),
                   showfliers=False)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=15)
    
    for i in range(3, len(positions), 3):
     ax.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)


    # legend
    custom_lines = [plt.Line2D([0], [0], color='dodgerblue', lw=4),
                    plt.Line2D([0], [0], color='green', lw=4),
                    plt.Line2D([0], [0], color='orange', lw=4)]
    ax.grid(alpha=0.3, axis='y')
    add_bottom_legend(fig, custom_lines, ['Classic', 'Flow', 'Mix'], fontsize=15)
    ax.set_xlabel('Proposal', fontsize=16)
    ax.set_ylabel('MCMC acceptances', fontsize=16)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if save == True:
        plt.savefig('boxplot_acceptances.pdf')
    plt.show()

###############################################################################

proposals = ['ind_1_hit', '1_hit', 'naive', 'r_hit_er', 'multi_r_hit_er']
plot_grid(['eps_time', 'eps_it'], 'exploratory_epsilon_grid_plot.pdf', proposals, save=True)
plot_grid(['loss_time', 'loss_it', 'loss_eps'], 'exploratory_loss_grid_plot.pdf', proposals, save=True)
plot_grid(['acc_time', 'acc_eps'], 'exploratory_acceptance_grid_plot.pdf', proposals, save=True)
plot_grid(['time_it'], 'exploratory_time_grid_plot.pdf', proposals, save=True)

#plot_time_boxplots(data = histories, save = True)
#plot_acceptances_boxplots(data = histories, save = True)

  

###############################################################################

