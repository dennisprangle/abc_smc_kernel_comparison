import torch
import numpy as np
from algorithms.utils import wdist, save_summary
from pathlib import Path
from nflows.distributions import StandardNormal
from models.MG1 import MG1Model
from algorithms.utils import norm_to_unif
import pandas as pd
import argparse
from wrapt_timeout_decorator import timeout
from algorithms.mcmc import KERNEL_NAMES

args_parser = argparse.ArgumentParser(description='List options for the experiment')
args_parser.add_argument('--name', default='mg1', help='Name of the experiment')
args_parser.add_argument('--proposal', default='classic_ind', choices=['classic_ind', 'classic_rw', 'mix', 'flow'])
args_parser.add_argument('--mix_components', default=[5], type=int, nargs='+', help='Number of components in mixture proposal (use 2 arguments to search over a range)')
args_parser.add_argument('--max_mins', default=60., type=float, help='Stop ABC after exceeding this runtime')
args_parser.add_argument('--erp', default=0.5, type=float, help='Epsilon reduction parameter (target fraction of unique alive particles)')
args_parser.add_argument('--n_particles', default=1000, type=int, help='Number of particles (with fallback enabled, 90%% are used for training and 10%% for testing)')
args_parser.add_argument('--augment', action='store_true', help='Augment parameters with latent variables')
args_parser.add_argument('--quantile_summaries', action='store_true', help='Use quantile summary statistics')
args_parser.add_argument('--seed', type=int, help='Random seed')
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

# Same observations as DIS paper
obs = torch.tensor(
    [ 4.67931388, 33.32367159, 16.1354178 ,  4.26184914, 21.51870177,
     19.26768645, 17.41684327,  4.39394293,  4.98717158,  4.00745068,
     17.13184198,  4.64447435, 12.10859597,  6.86436748,  4.199275  ,
     11.70312317,  7.06592802, 16.28106949,  8.66159665,  4.33875566])

model = MG1Model(nobs=obs.shape[0])

if args.quantile_summaries:
    summarise = lambda x : torch.quantile(x, torch.tensor([0., 0.25, 0.5, 0.75, 1.]), dim=-1).transpose(0,-1)
else:
    summarise = lambda x : x

obs = summarise(obs)

if args.augment:
    # rows of `inputs` should be augmented parameters
    ninputs = 43
    simulator = lambda inputs : summarise(model.simulator(inputs))
else:
    # rows of `inputs` should be ordinary parameters
    ninputs = 3
    simulator = lambda inputs : summarise(model.simulate_from_parameters(inputs))

prior = StandardNormal(shape=[ninputs])

if args.proposal == 'classic_ind' or args.proposal == 'classic_rw':
    from algorithms.abc_smc import ABC_SMC
    abc = ABC_SMC(
        N=args.n_particles,
        simulator = simulator,
        n_inputs = ninputs,
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
        simulator = simulator,
        n_inputs = ninputs,
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
        N_train= N_train,
        N_test = N_test,
        simulator = simulator,
        n_inputs = ninputs,
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
abc_samp[:,0] = norm_to_unif(abc_samp[:,0], 0, 1./3.)
abc_samp[:,1] = norm_to_unif(abc_samp[:,1], 0, 10.)
abc_samp[:,2] = norm_to_unif(abc_samp[:,2], 0, 10.)
mcmc_samp =  np.loadtxt('ref_samples/mg1_ref_sample.txt')
wloss = wdist(abc_samp, mcmc_samp)

model_name = 'MG1'
if args.quantile_summaries:
    model_name += '_quantile'

summary = {
    'model': model_name,
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
