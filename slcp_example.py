import torch
import numpy as np
from algorithms.utils import wdist, save_summary
from pathlib import Path
from nflows.distributions import StandardNormal
import matplotlib.pyplot as plt
from algorithms.utils import norm_to_unif
import pandas as pd
import argparse
from wrapt_timeout_decorator import timeout
from algorithms.mcmc import KERNEL_NAMES

args_parser = argparse.ArgumentParser(description='List options for the experiment')
args_parser.add_argument('--name', default='slcp', help='Name of the experiment')
args_parser.add_argument('--proposal', default='classic_ind', choices=['classic_ind', 'classic_rw', 'mix', 'flow'])
args_parser.add_argument('--mix_components', default=[5], type=int, nargs='+', help='Number of components in mixture proposal (use 2 arguments to search over a range)')
args_parser.add_argument('--max_mins', default=60., type=float, help='Stop ABC after exceeding this runtime')
args_parser.add_argument('--erp', default=0.5, type=float, help='Epsilon reduction parameter (target fraction of unique alive particles)')
args_parser.add_argument('--n_particles', default=1000, type=int, help='Number of particles (with fallback enabled, 90%% are used for training and 10%% for testing)')
args_parser.add_argument('--seed', type=int, help='Random seed')
args_parser.add_argument('--show_plots', action='store_true', help='Show plots during algorithm')
args_parser.add_argument('--save_history', action='store_true', help='Save history from algorithm')
args_parser.add_argument('--disable_fallback', action='store_true', help='Disable fallback to classic proposal')
args_parser.add_argument('--kernel', default='ind_1_hit', choices=KERNEL_NAMES, help='MCMC kernel')
args_parser.add_argument('--hits', default=2, type=int, help='Number of hits required in kernel')
args_parser.add_argument('--flow', default='spline', choices=['spline', 'spline_two_layers', 'coupling'], help='Flow type for flow proposal')
args_parser.add_argument('--target_prev', action='store_true', help='Train proposal on previous target distribution')
args_parser.add_argument('--defensive_proportion', default=0., type=float, help='Defensive proposal proportion')
args_parser.add_argument('--results_dir', help='Save summary to a new file in this directory (for parallel runs) instead of appending to results/results.csv')

args = args_parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if len(args.mix_components) == 1:
    min_mixture_components = max_mixture_components = args.mix_components[0]
elif len(args.mix_components) == 2:
    min_mixture_components, max_mixture_components = args.mix_components
else:
    raise ValueError('Invalid number of mixture components')

train_target = 'previous' if args.target_prev else 'current'

# Split of particles into training and test sets when fallback is enabled (900/100 by default)
n_test = args.n_particles // 10
n_train = args.n_particles - n_test

@torch.no_grad()
def slcp(theta):
    """Simulator for the 'simple likelihood complex posterior' example

    `theta` should be a (n,5) shape tensor whose rows are parameter vectors
    
    The parameters should be from standard normal distributions, and will be transformed to uniform draws.
    """
    theta = norm_to_unif(theta, -3., 3.)
    n = theta.shape[0]
    s1 = theta[:,2]**2.
    s2 = theta[:,3]**2.
    rho = torch.tanh(theta[:,4])
    eps = 1e-6
    m = theta[:,0:2]
    S = torch.empty([n,2,2])
    S[:,0,0] = s1**2. + eps
    S[:,0,1] = rho*s1*s2
    S[:,1,0] = rho*s1*s2
    S[:,1,1] = s2**2. + eps
    sample = torch.distributions.MultivariateNormal(m, S).sample([4])
    return sample.transpose(0,1).reshape((n,-1))

prior = StandardNormal(shape=[5])

#Observations simulated under the parameters (0.7, -2.9, -1, -0.9, 0.6) from the original paper
obs = torch.tensor([
  1.3614, -2.4299,
  0.7617, -2.4486,
  0.2481, -3.2101,
  -0.8228, -3.3016
])

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if args.proposal == 'classic_ind' or args.proposal == 'classic_rw':
    from algorithms.abc_smc import ABC_SMC
    abc = ABC_SMC(
        N=args.n_particles,
        simulator= slcp,
        n_inputs = 5,
        observations=obs,
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
        N_train = n_train,
        N_test = n_test,
        simulator = slcp,
        n_inputs = 5,
        observations=obs,
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
    N_train = n_train
    N_test = n_test
    if args.disable_fallback:
        N_train = args.n_particles
        N_test = 0
    abc = TransportABCMix(
        N_train = N_train,
        N_test = N_test,
        simulator = slcp,
        n_inputs = 5,
        observations = obs,
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
        if max_time <= 0:
            break
        train = timeout(max_time)(abc.training_iteration)
        train()
    except (TimeoutError):
        break
    if args.show_plots:
        plt.figure('Samples')
        plot_samp,_ = abc.particles['train'].sample(200)
        plot_samp = plot_samp.detach().numpy()
        plot_samp = norm_to_unif(plot_samp, -3., 3.)
        plt.plot(plot_samp[:,2], plot_samp[:,3], "o", alpha=0.7)
        if args.proposal == 'mix':
            plt.figure(f'components_{abc.iteration}')
            samples, sources = abc.mixture_proposal.mix.model_.sample(1000)
            samples = norm_to_unif(samples, -3., 3.)
            for i in range(5):
                filter = (sources==i)
                plot_samp = samples[filter]
                plt.plot(plot_samp[:,2], plot_samp[:,3], "o", alpha=0.7)
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
        plt.pause(0.1)

training_proportion = abc.training_proportion

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

# Wasserstein distance to MCMC samples
mcmc_samp =  np.loadtxt('ref_samples/slcp_ref_sample.txt')
abc_samp = norm_to_unif(abc_samp, -3., 3.)
wloss = wdist(abc_samp, mcmc_samp)

summary = {
    'model': 'slcp',
    'proposal': args.proposal,
    'kernel': args.kernel,
    'flow': args.flow,
    'min_mixture_components': min_mixture_components,
    'max_mixture_components': max_mixture_components,
    'fallback': not args.disable_fallback,
    'train_target': train_target,
    'hits': args.hits,
    'defensive_proportion': args.defensive_proportion,
    'n_particles': args.n_particles,
    'erp': args.erp,
    'seed': args.seed,
    'final_epsilon': eps_final,
    'epsilon_at_half_time': eps_half,
    'loss': wloss,
    'training_proportion': training_proportion
}
Path('results').mkdir(exist_ok=True)
save_summary(summary, args.results_dir)

if args.save_history:
    df = pd.DataFrame.from_dict(abc.history)
    df.to_csv(f'results/{args.name}_history.csv')

if args.show_plots:
    input('Press enter to exit')
