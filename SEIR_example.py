import torch
import numpy as np
from algorithms.utils import wdist
from pathlib import Path
from models.epimodels import SEIRModel
import pandas as pd
from os.path import exists
import argparse
from wrapt_timeout_decorator import timeout

args_parser = argparse.ArgumentParser(description='List options for the experiment')
args_parser.add_argument('--name', default='seir', help='Name of the experiment')
args_parser.add_argument('--proposal', default='classic_ind', choices=['classic_ind', 'classic_rw', 'mix', 'flow'])
args_parser.add_argument('--mix_components', default=[5], type=int, nargs='+', help='Number of components in mixture proposal (use 2 arguments to search over a range)')
args_parser.add_argument('--max_mins', default=60., type=float, help='Stop ABC after exceeding this runtime')
args_parser.add_argument('--erp', default=0.5, type=float, help='Epsilon reduction parameter (target fraction of unique alive particles)')
args_parser.add_argument('--seed', type=int, help='Random seed')
args_parser.add_argument('--show_plots', action='store_true', help='Show plots during algorithm')
args_parser.add_argument('--save_history', action='store_true', help='Save history from algorithm')
args_parser.add_argument('--disable_fallback', action='store_true', help='Disable fallback to classic proposal')
args_parser.add_argument('--kernel', default='ind_1_hit', choices=['naive', '1_hit', 'ind_1_hit', 'r_hit', 'multi_r_hit'], help='MCMC kernel')
args_parser.add_argument('--hits', default=2, type=int, help='Number of hits required in kernel')
args_parser.add_argument('--flow', default='spline', choices=['spline', 'spline_two_layers', 'coupling'], help='Flow type for flow proposal')
args_parser.add_argument('--target_prev', action='store_true', help='Train proposal on previous target distribution')
args_parser.add_argument('--defensive_proportion', default=0., type=float, help='Defensive proposal proportion')

args = args_parser.parse_args()

if len(args.mix_components) == 1:
    min_mixture_components = max_mixture_components = args.mix_components[0]
elif len(args.mix_components) == 2:
    min_mixture_components, max_mixture_components = args.mix_components
else:
    raise ValueError('Invalid number of mixture components')

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

train_target = 'previous' if args.target_prev else 'current'

model = SEIRModel(max_time=30, time_step=1, E0=10, S0=990)
simulator = lambda inputs : model.simulate(torch.exp(inputs)) # inputs rows should be parameters (on log scale)
                                                              #inputs are recovery rate and infection rate
obs = torch.tensor([0, 3, 0, 1, 4, 1, 0, 11, 8, 11, 17, 32, 32, 32, 49, 41, 46, 55, 39, 38, 34, 14, 16, 7, 0, 1, 2, 0, 1, 0])
ninputs = 3

prior = torch.distributions.Normal(
    loc=torch.tensor([-3.,-1.,-0.5 ]),
    scale=torch.tensor([2.,2., 2.])
)
prior = torch.distributions.Independent(prior, 1)

if args.proposal == 'classic_ind' or args.proposal == 'classic_rw':
    from algorithms.abc_smc import ABC_SMC
    abc = ABC_SMC(
        N=1000,
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
        N_train = 900,
        N_test = 100,
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
    N_train = 900
    N_test = 100
    if args.disable_fallback:
        N_train = 1000
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
        defensive_proportion=args.defensive_proportion
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
pmmh_samp =  np.loadtxt('ref_samples/seir_ref_sample.txt')
wloss = wdist(abc_samp, pmmh_samp[:,[2,1,0]])


summary = {
    'model': 'seir',
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

np.savetxt(f'results/seir_abc_sample.txt', abc_samp)

# plt.style.use('ggplot')
# plt.figure()
# plt.hist(abc_samp[:,0], density=True)
# plt.hist(pmmh_samp[:,2], density=True, alpha=0.5)
# plt.axvline(x=-3, color='r')
# plt.xlabel('$log \\gamma$')

# plt.figure()
# plt.hist(abc_samp[:,1], density=True)
# plt.hist(pmmh_samp[:,1], density=True, alpha=0.5)
# plt.axvline(x=-1, color='r')
# plt.xlabel('$log \\beta$')

# plt.figure()
# plt.hist(abc_samp[:,2], density=True)
# plt.hist(pmmh_samp[:,0], density=True, alpha=0.5)
# plt.axvline(x=-0.5, color='r')
# plt.xlabel('$log \\alpha$')
