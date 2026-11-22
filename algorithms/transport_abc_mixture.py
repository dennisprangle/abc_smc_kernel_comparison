from time import time
import torch
from algorithms.abc_smc import ABC_SMC
from algorithms.proposals import ClassicInd, DefensiveProposal, MixtureProposal

class TransportABCMix(ABC_SMC):
    """Transport ABC SMC

    `N_train` is number of training particles
    `N_test` is number of test particles
    `simulator` is a function mapping inputs to simulated data
    `n_inputs` is total number of inputs for the simulator
    `observations` is the observed dataset (or summaries of it)
    `prior` is the prior distribution (as a torch or nflows distribution)
    `epsilon_reduction_parameter` controls how epsilon is reduced
    `get_distance` is a function mapping 2 datasets to a distance (Euclidean distance by default)
    `r` is the value of r to use in the r-hit kernel
    `min_mixture_components` is the minimum number of mixture components to consider
    `max_mixture_components` is the maximum number of mixture components to consider
    `enable_fallback` is whether to use classic ABC when it has a better test loss
    `kernel_name` indicates which MCMC kernel to use
    `train_target` defines the target distribution when training the flow - options are 'current' and 'previous'
    `defensive_proportion` sets up a defensive proposal if given a positive value
    """
    def __init__(
            self, N_train, N_test, simulator, n_inputs, observations, prior,
            epsilon_reduction_parameter=0.5,
            get_distance = lambda y1, y2: torch.norm(y1-y2, 2, dim=1),
            r=2,
            min_mixture_components = 5,
            max_mixture_components = 5,
            enable_fallback = True,
            kernel_name = 'ind_1_hit',
            train_target = 'current',
            defensive_proportion = 0.
    ):
        self.init_common(
            simulator, n_inputs, observations, prior, 'mixture', # proposal must be mixture
            epsilon_reduction_parameter, get_distance, r, kernel_name, train_target, defensive_proportion
        )
        self.min_mixture_components = min_mixture_components
        self.max_mixture_components = max_mixture_components
        self.self_label_init = (min_mixture_components == max_mixture_components)
        print('Initialising particles')
        if enable_fallback:
            self.particles = {
                'train':self.init_particles(N_train),
                'test':self.init_particles(N_test)
            }
        else:
            self.particles = {
                'train':self.init_particles(N_train)
                # No need for test particles if fallback disabled
            }
        self.enable_fallback = enable_fallback
        self.nr_fallback = 0

    def loss(self, proposal, pars, weights):
        """Loss (weighted cross entropy) under `proposal` for given parameters and weights"""
        loss = -proposal.log_prob(pars)
        if self.train_target == 'current':
            # For current target apply weights (i.e. only use particles with epsilon below threshold)
            loss = loss * weights
        return loss.mean() # Mean not sum so different sized inputs comparable

    def training_iteration(self):
        print(f'\nStarting ABC SMC iteration {self.iteration+1:d}')
        u = torch.rand(1) # Random sample for systemic resampling
        eps_all = [
            self.get_eps(p, u=u)
            for p in self.particles.values()
        ]
        self.epsilon = max(eps_all)
        print(f'New epsilon is {self.epsilon:.2e}')
        for p in self.particles.values():
            p.reweight(self.epsilon)
        with self.training_timer():
            self.mixture_proposal = MixtureProposal(
                self.particles['train'],
                min_components=self.min_mixture_components,
                max_components=self.max_mixture_components,
                safe_label_init=self.self_label_init,
                use_weights = (self.train_target == 'current')
            )
        self.kernel.proposal = self.mixture_proposal
        if self.defensive_proportion > 0:
            self.kernel.proposal = DefensiveProposal(
                self.prior,
                self.kernel.proposal,
                defensive_proportion = self.defensive_proportion
            )
        self.kernel.epsilon = self.epsilon
        if self.enable_fallback:
            with self.training_timer():
                mixture_test_loss = self.loss(
                    self.mixture_proposal,
                    self.particles['test'].values,
                    self.particles['test'].weights
                ).item()
                self.classic_proposal = ClassicInd(
                    self.particles['train'],
                    use_weights = (self.train_target == 'current')
                )
                classic_test_loss = self.loss(
                    self.classic_proposal,
                    self.particles['test'].values,
                    self.particles['test'].weights,
                )
            if mixture_test_loss > classic_test_loss:
                print('Falling back to classic ABC proposal!')
                self.nr_fallback +=1
                self.kernel.proposal = self.classic_proposal

        old_total_sims = self.total_sims
        for p in self.particles.values():
            p.resample(method='systematic', u=u) # Resample
        mcmc_updates, acc_count = self.make_mcmc_moves() # Updates self.particles
        self.iteration += 1
        elapsed_time = self.elapsed_time
        new_history = {
            'iteration': self.iteration,
            'epsilon': self.epsilon,
            'time': elapsed_time,
            'training_proportion': self.training_time / elapsed_time,
            'unique_count_train': self.particles['train'].unique_count(),
            'mcmc_acceptances': acc_count,
            'nr fallbacks': self.nr_fallback
        }
        self.history_list.append(new_history)
        self._history_cached = False
        print(f'End of iteration, elapsed time (mins) {elapsed_time/60.:.1f}')
        print(f'Sims this iteration {self.total_sims - old_total_sims:d}')
        print(f'MCMC acceptances {acc_count:d} / {mcmc_updates:d}')
