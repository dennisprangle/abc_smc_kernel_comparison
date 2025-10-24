import torch
import numpy as np
from algorithms.utils import wdist
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import argparse
from wrapt_timeout_decorator import timeout

args_parser = argparse.ArgumentParser(description='List options for the experiment')
args_parser.add_argument('--name', default='gaussian_mixture', help='Name of the experiment')
args_parser.add_argument('--proposal', default='classic_ind', choices=['classic_ind', 'classic_rw', 'mix', 'flow'])
args_parser.add_argument('--mix_components', default=[5], type=int, nargs='+', help='Number of components in mixture proposal (use 2 arguments to search over a range)')
args_parser.add_argument('--max_mins', default=60., type=float, help='Stop ABC after exceeding this runtime')
args_parser.add_argument('--erp', default=0.5, type=float, help='Epsilon reduction parameter (target fraction of unique alive particles)')
args_parser.add_argument('--seed', type=int, help='Random seed')
args_parser.add_argument('--obs', default=0., type=float, help='Observed data')
args_parser.add_argument('--show_plots', action='store_true', help='Show plots during algorithm')
args_parser.add_argument('--save_plots', action='store_true', help='Save samples plots after algorithm')
args_parser.add_argument('--save_history', action='store_true', help='Save history from algorithm')
args_parser.add_argument('--disable_fallback', action='store_true', help='Disable fallback to classic proposal')
args_parser.add_argument('--kernel', default='ind_1_hit', choices=['naive', '1_hit', 'ind_1_hit', 'r_hit', 'multi_r_hit'], help='MCMC kernel')
args_parser.add_argument('--hits', default=2, type=int, help='Number of hits required in kernel')
args_parser.add_argument('--flow', default='spline', choices=['spline', 'spline_two_layers', 'coupling'], help='Flow type for flow proposal')
args_parser.add_argument('--target_prev', action='store_true', help='Train proposal on previous target distribution')
args_parser.add_argument('--defensive_proportion', default=0., type=float, help='Defensive proposal proportion')

args = args_parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

do_plots = args.save_plots or args.show_plots
if args.save_plots:
    Path('plots').mkdir(exist_ok=True)
if args.show_plots:
    plt.ion()

if len(args.mix_components) == 1:
    min_mixture_components = max_mixture_components = args.mix_components[0]
elif len(args.mix_components) == 2:
    min_mixture_components, max_mixture_components = args.mix_components
else:
    raise ValueError('Invalid number of mixture components')

train_target = 'previous' if args.target_prev else 'current'

@torch.no_grad()
def gaussian_mixture(theta):
    """Simulator for Gaussian mixture example from Sisson et al 2007

    `theta` is a shape (n) or (n,1) tensor whose rows are location parameters

    The simulator output is uniformly randomly selected from z, 0.1z where z~N(0,1).
    """
    if len(theta.shape) == 2:
        theta.squeeze_(1)
    size = theta.shape[0]
    scales = torch.tensor([0.1, 1]).multinomial(
        size, replacement=True
    )
    samples = theta + torch.randn(size) * scales
    return samples.unsqueeze(1)

# validate_args=False means log_prob outside support is -inf rather than error
# This is desired behaviour for MCMC proposals outside the support
prior = torch.distributions.Uniform(-10, 10, validate_args=False)

if args.proposal == 'classic_ind' or args.proposal == 'classic_rw':
    from algorithms.abc_smc import ABC_SMC
    abc = ABC_SMC(
        N=1000,
        simulator= gaussian_mixture,
        n_inputs = 1,
        observations=args.obs,
        prior=prior,
        proposal=args.proposal,
        epsilon_reduction_parameter=args.erp,
        kernel_name=args.kernel,
        r=args.hits,
        train_target=train_target,
        defensive_proportion = args.defensive_proportion
    )
elif args.proposal == 'flow':
    from algorithms.transport_abc import TransportABC
    abc = TransportABC(
        N_train = 900,
        N_test = 100,
        simulator = gaussian_mixture,
        n_inputs = 1,
        observations=args.obs,
        prior=prior,
        epsilon_reduction_parameter=args.erp,
        transform_name=args.flow,
        enable_fallback=not args.disable_fallback,
        kernel_name=args.kernel,
        r=args.hits,
        train_target=train_target,
        defensive_proportion = args.defensive_proportion
    )
    abc.pretrain()
elif args.proposal == 'mix':
    from algorithms.transport_abc_mixture import TransportABCMix
    N_train = 900
    N_test = 100
    if args.disable_fallback:
        N_train = 1000
        N_test = 0
    abc = TransportABCMix(
        N_train = N_train,
        N_test = N_test,
        simulator = gaussian_mixture,
        n_inputs = 1,
        observations = args.obs,
        prior = prior,
        epsilon_reduction_parameter = args.erp,
        min_mixture_components = min_mixture_components,
        max_mixture_components = max_mixture_components,
        enable_fallback=not args.disable_fallback,
        kernel_name=args.kernel,
        r=args.hits,
        train_target=train_target,
        defensive_proportion = args.defensive_proportion
    )
else:
    raise ValueError('Unknown proposal')

while True:
    # abc_samp will be particles from last iteration *before* max time reached
    abc_samp = abc.particles['train'].values.detach().numpy()
    try:
        max_time = 60.*args.max_mins - abc.elapsed_time
        train = timeout(max_time)(abc.training_iteration)
        train()
    except (TimeoutError):
        break
    if do_plots: #TODO: update to 1d plots
        plt.figure('Samples')
        plot_samp,_ = abc.particles['train'].sample(200)
        plot_samp = plot_samp.detach().numpy()
        plt.hist(plot_samp, density=True, alpha=0.5)
    if args.proposal == 'mix' and args.show_plots:
        # The user can save these manually if desired
        plt.figure(f'components_{abc.iteration}')
        samples, sources = abc.mixture_proposal.mix.model_.sample(1000)
        for i in range(sources.max()+1):
            filter = (sources==i)
            plt.hist(samples[filter], density=True, alpha=0.5)
    if args.show_plots:
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
        plt.pause(0.1)

if args.save_plots:
    plt.figure('Samples')
    plt.savefig(f'plots/{args.name}_samples.pdf')

eps_final = np.inf
eps_half = np.inf # Will hold final epsilon value *before* half max time reached
h = abc.history
if abc.history is not None:
    eps_array = np.array(h['epsilon'])
    if len(eps_array) > 0:
        eps_final = eps_array[-1]

    t_array = np.array(abc.history['time'])
    i = np.where(t_array < 30. * args.max_mins)[0]
    if len(i) > 0:
        i = i[-1]
        eps_half = eps_array[i]

# Wasserstein distance to reference sample
ref_samp =  np.loadtxt('ref_samples/gaussian_mixture_ref_sample.txt')
ref_samp = np.expand_dims(ref_samp, 1)
wloss = wdist(abc_samp, ref_samp)

summary = {
    'model': 'gaussian_mixture',
    'proposal': args.proposal,
    'kernel': args.kernel,
    'flow': args.flow,
    'min_mixture_components': min_mixture_components,
    'max_mixture_components': max_mixture_components,
    'fallback': not args.disable_fallback,
    'train_target': train_target,
    'final_epsilon': eps_final,
    'epsilon_at_half_time': eps_half,
    'loss': wloss
}
df = pd.DataFrame.from_dict([summary]) # See https://stackoverflow.com/a/63972829
Path('results').mkdir(exist_ok=True)
output_path='results/results.csv'
df.to_csv(output_path, mode='a', header=not exists(output_path))

if args.save_history:
    df = pd.DataFrame.from_dict(abc.history)
    df.to_csv(f'results/{args.name}_history.csv')

if args.show_plots:
    input('Press enter to exit')
