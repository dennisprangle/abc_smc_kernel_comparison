import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import argparse

args_parser = argparse.ArgumentParser(description='List options for comparison plots')
args_parser.add_argument('filenames', nargs='+', help='CSV files to plot')
args_parser.add_argument('--legend', nargs='+', help='Legend labels for each file')
args = args_parser.parse_args(['classicind_ind_1_hit_history.csv',
                               'mix_ind_1_hit_history.csv', 
                               'flow_ind_1_hit_history.csv',
                               'classicrw_1_hit_history.csv',
                               'classicind_1_hit_history.csv', 
                               'mix_1_hit_history.csv', 
                               'flow_1_hit_history.csv',
                               'classicrw_naive_history.csv',
                               'classicind_naive_history.csv',
                               'mix_naive_history.csv', 
                               'flow_naive_history.csv',
                               'classicrw_multi_r_hit_history.csv', 
                               'classicind_multi_r_hit_history.csv',
                               'mix_multi_r_hit_history.csv', 
                               'flow_multi_r_hit_history.csv',])

names = {
    'naive': 'ABC-MH',
    '1_hit': 'One-hit',
    'ind_1_hit': 'Independence one-hit',
    'r_hit': 'Single proposal r-hit',
    'multi_r_hit': 'Multiple proposal r-hit'
}

histories = {}
cols = {'classicind': 'mediumblue','classicrw': 'royalblue', 'flow': 'teal', 'mix': 'mediumvioletred'}

#TODO: remove?
mark = {'classicind': '-o','classicrw': '-o', 'flow': '-o', 'mix': '-o'}

for fname in args.filenames:
    df = pd.read_csv(fname)
    name = fname.split('_history')[0] # remove "_history.csv"
    histories[name]=df



plt.ion()

#plot time versus iteration - different kernels for a given proposal
def plot_time_vs_it(proposal, data = histories, titles = names, colors=cols, markers=mark, save = False):
    list_legend = []
    plt.figure(figsize=(6, 4))
    for name, hist in data.items():    
        if proposal == "_".join(name.split("_")[1:]):
            plt.plot(np.array(hist['iteration']), np.array(hist['time']), markers[name.split('_')[0]], linewidth=2, color=colors[name.split('_')[0]])
            list_legend.append(name.split('_')[0])
    plt.yscale('log')
    plt.ylabel('Time(s)')
    plt.ylim(1e0, 1e4)
    plt.xlabel('Iteration', fontsize=15)
    plt.title(titles[proposal], fontsize=16)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(list_legend)
    if save == True:
        plt.savefig(f'figures/{proposal}_timeVSit.pdf')
    plt.show()


#plot epsilon versus time and iteration  -  different kernels for a given proposal     
def plot_eps(proposal,data = histories, titles = names, colors=cols, markers=mark, save = False):
    list_legend = []
    fig, (axs0, axs1) = plt.subplots(1, 2, figsize=(10, 5))
    for name, hist in data.items():    
        if proposal == "_".join(name.split("_")[1:]):
            axs0.plot(np.array(hist['time']), np.array(hist['epsilon']), markers[name.split('_')[0]], linewidth=2, color=colors[name.split('_')[0]] )
            axs1.plot(np.array(hist['iteration']), np.array(hist['epsilon']), markers[name.split('_')[0]], linewidth=2, color=colors[name.split('_')[0]])
            axs0.set(xlabel='Time (s)', yscale='log')
            axs1.set(xlabel='Iterations', yscale='log')
            list_legend.append(name.split('_')[0])
    axs0.set_ylim(1e-6, 1.5*1e0)
    axs1.set_ylim(1e-6, 1.5*1e0)
    axs1.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(titles[proposal], fontsize=16)
    fig.legend(list_legend, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=4)
    fig.text(0.08, 0.5, '$\\epsilon$', va='center', rotation='vertical', fontsize=15)
    fig.tight_layout(rect=[0.1, 0.1, 1, 0.95])
    if save == True:
        plt.savefig(f'figures/{proposal}_epsilon.pdf')
    plt.show()
    
#plot loss versus time and iteration  -  different kernels for a given proposal      
def plot_loss(proposal,data = histories, titles = names, colors=cols, markers=mark, save = False):
    list_legend = []
    fig, (axs0, axs1) = plt.subplots(1, 2, figsize=(10, 5))
    for name, hist in data.items():    
        if proposal == "_".join(name.split("_")[1:]):
            axs0.plot(np.array(hist['time']), np.array(hist['wloss']), markers[name.split('_')[0]], linewidth=2, color=colors[name.split('_')[0]])
            axs1.plot(np.array(hist['iteration']), np.array(hist['wloss']),  markers[name.split('_')[0]], linewidth=2, color=colors[name.split('_')[0]])
            axs0.set(xlabel='Time (s)', yscale='log')
            axs1.set(xlabel='Iterations', yscale='log')
            list_legend.append(name.split('_')[0])
    fig.suptitle(titles[proposal], fontsize=16)
    fig.legend(list_legend, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=4)
    fig.text(0.08, 0.5, 'Wassertein loss', va='center', rotation='vertical', fontsize=15)
    fig.tight_layout(rect=[0.1, 0.1, 1, 0.95])
    if save == True:
        plt.savefig(f'figures/{proposal}_loss.pdf')
    plt.show()
 
#plot mcmc acceptances versus time and iteration  -  different kernels for a given proposal     
def plot_acc(proposal,data = histories, titles = names, colors=cols, markers=mark, save = False):
       list_legend = []
       plt.figure(figsize=(6, 4))
       for name, hist in data.items():    
           if proposal == "_".join(name.split("_")[1:]):
               plt.plot(np.array(hist['time']), np.array(hist['mcmc_acceptances'])/1000, markers[name.split('_')[0]], linewidth=2, color=colors[name.split('_')[0]])
               list_legend.append(name.split('_')[0])
       plt.title(titles[proposal], fontsize=16)
       plt.xlabel('Time (s)', fontsize=15)
       plt.ylim(0., 1.)
       plt.legend(list_legend)
       plt.ylabel('MCMC acceptance rate', fontsize=15)
       if save == True:
           plt.savefig(f'figures/{proposal}_acc.pdf')
       plt.show()
       
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
    ax.legend(custom_lines, ['Classic', 'Flow', 'Mix'],fontsize=15)
    ax.set_xlabel('Proposal', fontsize=16)
    ax.set_ylabel('Time(s)', fontsize=16)

    plt.tight_layout()
    if save == True:
        plt.savefig('figures/boxplot_time.pdf')
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
    ax.legend(custom_lines, ['Classic', 'Flow', 'Mix'], fontsize = 15)
    ax.set_xlabel('Proposal', fontsize=16)
    ax.set_ylabel('MCMC acceptances', fontsize=16)

    plt.tight_layout()
    if save == True:
        plt.savefig('figures/boxplot_acceptances.pdf')
    plt.show()

###############################################################################



for proposal in ['ind_1_hit', '1_hit', 'naive', 'multi_r_hit']:
    plot_eps(proposal, data = histories, save = True) 
    plot_acc(proposal, data = histories, save = True) 
#    plot_loss(proposal, data = histories, save = True) 
    plot_time_vs_it(proposal, data = histories, save = True) 

#plot_time_boxplots(data = histories, save = True)
#plot_acceptances_boxplots(data = histories, save = True)

  

###############################################################################

