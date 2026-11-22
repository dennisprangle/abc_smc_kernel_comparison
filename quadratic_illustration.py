## Produce illustrative plots of the various proposals

import torch
import numpy as np
from pathlib import Path
from nflows.distributions import StandardNormal
from algorithms.transport_abc import TransportABC
from algorithms.proposals import ClassicInd, MixtureProposal
import matplotlib as mpl
import matplotlib.pyplot as plt
# Panels are drawn at the size they appear in the paper (0.19 of the text width, about 1.4in),
# so these are the sizes printed on the page. Tick text matches the analysis plots
PANEL_SIZE = 1.4  # inches
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['axes.linewidth'] = 0.6

def new_panel(name):
    """Start a small square figure with the shared axis limits, ticks and faint grid"""
    fig = plt.figure(name, figsize=(PANEL_SIZE, PANEL_SIZE))
    ax = fig.gca()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.grid(alpha=0.3)
    return fig

def save_panel(fig, filename):
    fig.tight_layout(pad=0.3)
    # No creation date, so an unchanged plot gives an identical file
    fig.savefig(filename, metadata={'CreationDate': None})

np.random.seed(1)
torch.manual_seed(1)

Path('plots').mkdir(exist_ok=True)
plt.ion()

@torch.no_grad()
def quadratic(theta):
    """Simulator for a quadratic example

    `theta` should be a (n,2) shape tensor whose rows are parameter pairs

    The simulator output is y = theta1 - theta2^2 + 0.01 z where z~N(0,1).
    So the posterior is close to the quadratic curve theta1 = y + theta2^2.
    The motivation is that this is sufficiently non-Gaussian for the standard ABC-PMC proposal to be a poor match,
    while transport ABC's proposal is a much better match.
    """
    size = theta.shape[0]
    epsilon = StandardNormal(shape=[1]).sample(size)*0.01
    sample = theta[:,0]-theta[:,1]**2+epsilon.view(-1)
    return sample.view(size,1)

prior = StandardNormal(shape=[2])

# Import MCMC samples
mcmc_samp =  np.loadtxt('ref_samples/quadratic_ref_sample.txt')   

# Run ABC
abc_flow = TransportABC(
    N_train = 900,
    N_test = 100,
    simulator = quadratic,
    n_inputs = 2,
    observations = 0.,
    prior = prior,
    epsilon_reduction_parameter = 0.5,
    enable_fallback = False,
    kernel_name="1_hit",
    train_target="current"
)

abc_flow.pretrain()
for i in range(3):
    abc_flow.training_iteration()

samp_abc,_ = abc_flow.particles['train'].sample(200)
samp_abc = samp_abc.detach().numpy()

prop_mix = MixtureProposal(abc_flow.particles['train'])
samp_mix, mix_sources = prop_mix.mix.model_.sample(1000)

prop_classic = ClassicInd(abc_flow.particles['train'])
samp_classic = [prop_classic.sample_and_log_prob()[0].detach().numpy() for _ in range(1000)]
samp_classic = np.vstack(samp_classic)

abc_flow.training_iteration() # A proposal flow is learned as a side effect
samp_flow = abc_flow.flow_proposal.flow.sample(1000).detach().numpy()

fig = new_panel('Mixture proposals')
for i in range(mix_sources.max()+1):
    filter = (mix_sources==i)
    x = samp_mix[filter]
    plt.plot(x[:,0], x[:,1], "o", markersize=2, alpha=0.3)
save_panel(fig, 'plots/illustration_prop_mix.pdf')

fig = new_panel('Flow proposals')
plt.plot(samp_flow[:,0], samp_flow[:,1], "o", markersize=2, alpha=0.3)
save_panel(fig, 'plots/illustration_prop_flow.pdf')

fig = new_panel('Classic proposals')
plt.plot(samp_classic[:,0], samp_classic[:,1], "o", markersize=2, alpha=0.3)
save_panel(fig, 'plots/illustration_prop_classic.pdf')

fig = new_panel('ABC samples')
plt.plot(samp_abc[:,0], samp_abc[:,1], "ok", markersize=2.5, alpha=1.)
save_panel(fig, 'plots/illustration_abc_samples.pdf')

fig = new_panel('MCMC samples')
plt.plot(mcmc_samp[:,0], mcmc_samp[:,1], ".b", markersize=4, alpha=1.)
save_panel(fig, 'plots/illustration_mcmc_samples.pdf')

plt.gcf().canvas.draw()
plt.gcf().canvas.flush_events()
plt.pause(0.1)

input('Press enter to exit')