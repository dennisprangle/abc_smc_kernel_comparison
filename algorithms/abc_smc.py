import numpy as np
from contextlib import contextmanager
from copy import deepcopy
from time import time
from pandas import DataFrame
import torch
import nflows.distributions as nfd
from algorithms.utils import Particles
from algorithms.mcmc import makeKernel
from algorithms.proposals import ClassicInd, ClassicRW, DefensiveProposal

class ABC_SMC:
    """Classic ABC SMC

    `N` is number of particles
    `simulator` is a function mapping inputs to simulated data
    `n_inputs` is total number of inputs for the simulator
    `observations` is the observed dataset (or summaries of it)
    `prior` is the prior distribution (as a torch or nflows distribution)
    `epsilon_reduction_parameter` controls how epsilon is reduced
    `get_distance` is a function mapping 2 datasets to a distance (Euclidean distance by default)
    `r` is the value of r to use in the r-hit kernel
    `kernel_name` indicates which MCMC kernel to use
    `train_target` defines the target distribution when training the flow - options are 'current' and 'previous'
    `defensive_proportion` sets up a defensive proposal if given a positive value
    """
    def __init__(
            self, N, simulator, n_inputs, observations, prior,
            proposal='classic_ind',
            epsilon_reduction_parameter=0.5,
            get_distance = lambda y1, y2: torch.norm(y1-y2, 2, dim=1),
            r=2,
            kernel_name = 'multi_r_hit',
            train_target = 'current',
            defensive_proportion = 0.
    ):
        self.init_common(
            simulator, n_inputs, observations, prior, proposal,
            epsilon_reduction_parameter, get_distance, r,
            kernel_name, train_target, defensive_proportion
        )
        print('Initialising particles')
        self.particles = {
            'train':self.init_particles(N)
        }

    def init_common(
        self, simulator, n_inputs, observations, prior, proposal,
        epsilon_reduction_parameter, get_distance, r, kernel_name, train_target, defensive_proportion
    ):
        """Parts of initialisation shared between this class and subclasses"""
        self.start_time = time()
        self.training_time = 0.
        self.simulator = simulator
        self.n_inputs = n_inputs
        self.observations = observations
        self.prior = prior
        self.proposal = proposal
        self.epsilon_reduction_parameter = epsilon_reduction_parameter
        self.get_distance = get_distance
        self.iteration = 0
        self.total_sims = 0
        self.epsilon = np.inf
        self.history_list = []
        self._history = None
        self._history_cached = True
        self.kernel = makeKernel(
            kernel_name = kernel_name,
            simulator = simulator,
            prior = prior,
            get_distance = get_distance,
            observations = observations,
            r = r
        )
        self.train_target = train_target
        self.defensive_proportion = defensive_proportion

    @property
    def history(self):
        if not self._history_cached:
            self._history = DataFrame(self.history_list)
            self._history_cached = True
        return self._history

    @property
    def elapsed_time(self):
        """Seconds since the algorithm started"""
        return time() - self.start_time

    @contextmanager
    def clock_paused(self):
        """Context manager: time spent inside it doesn't count towards `elapsed_time`"""
        pause_start_time = time()
        try:
            yield
        finally:
            self.start_time += time() - pause_start_time

    @contextmanager
    def training_timer(self):
        """Context manager: time spent inside it counts towards `training_time`"""
        timer_start_time = time()
        try:
            yield
        finally:
            self.training_time += time() - timer_start_time

    @property
    def training_proportion(self):
        """Proportion of `elapsed_time` spent training proposals"""
        return self.training_time / self.elapsed_time

    @torch.no_grad()
    def init_particles(self, nparticles):
        """Initialise particles, by performing simulations under the prior"""
        if isinstance(self.prior, nfd.Distribution):
            values = self.prior.sample(nparticles)
        else: # The other expected case is a torch distribution
            values = self.prior.sample([nparticles,])
        weights = torch.ones(nparticles)
        # Next line assumes vectorised simulator
        distances = self.get_distance(self.observations, self.simulator(values))
        return Particles(nparticles, values, weights, distances)

    @torch.no_grad()
    def get_eps(self, particles, u=None, quantile=False):
        """Return epsilon value based on given `particles`

        The default method is a bisection algorithm to get the required proportion of unique particles.
        Setting `quantile` to True instead uses a quantile of the distances.

        `u` is a single uniform random sample to be used in systematic resampling.
        If not supplied this will be generated.
        """
        if quantile:
            return torch.kthvalue(
                particles.distances,
                int(self.epsilon_reduction_parameter*particles.count)
            )[0].item()

        #Bisection algorithm
        target_unique = int(self.epsilon_reduction_parameter * particles.count)
        lower = 0.
        upper = self.epsilon
        if upper == np.inf:
            upper = particles.distances.max().item()
        n_unique = particles.unique_count(upper)
        i=0
        if n_unique < target_unique:
            return upper
        while i<50 and n_unique != target_unique:
            eps_guess = (lower + upper) / 2.
            new_particles = deepcopy(particles)
            new_particles.reweight(eps_guess)
            if sum(new_particles.weights) > 0:
                new_particles.resample(method='systematic', u=u)
                n_unique = new_particles.unique_count(eps_guess)
            else:
                #Can't resample if all weights are zero!
                n_unique = 0
            if n_unique >= target_unique:
                upper = eps_guess
            else:
                lower = eps_guess
            i += 1
        # Consider returning eps=0 if it's still an endpoint
        if lower == 0.:
            n_unique = particles.unique_count(0.)
            if n_unique >= target_unique:
                return 0.
        # Be conservative by returning upper end of remaining range
        return upper

    def training_iteration(self):
        print(f'\nStarting ABC SMC iteration {self.iteration+1:d}')
        p = self.particles['train']
        u = torch.rand(1) # Random sample for systemic resampling
        self.epsilon = self.get_eps(p, u=u)
        print(f'New epsilon is {self.epsilon:.2e}')
        p.reweight(self.epsilon)
        old_total_sims = self.total_sims
        with self.training_timer():
            if self.proposal == 'classic_ind':
                self.classic_proposal = ClassicInd(
                    self.particles['train'],
                    use_weights = (self.train_target == 'current')
                )
            elif self.proposal == 'classic_rw':
                self.classic_proposal = ClassicRW(
                    self.particles['train'],
                    use_weights = (self.train_target == 'current')
                )
            else:
                raise NotImplementedError('Proposal not implemented')
        p.resample(method='systematic', u=u) # Resample
        self.kernel.epsilon = self.epsilon
        self.kernel.proposal = self.classic_proposal
        if self.defensive_proportion > 0:
            self.kernel.proposal = DefensiveProposal(
                self.prior,
                self.kernel.proposal,
                defensive_proportion = self.defensive_proportion
            )
        mcmc_updates, acc_count = self.make_mcmc_moves() # Updates self.particles
        self.iteration += 1
        elapsed_time = self.elapsed_time
        new_history = {
            'iteration': self.iteration,
            'epsilon': self.epsilon,
            'time': elapsed_time,
            'training_proportion': self.training_time / elapsed_time,
            'unique_count_train': p.unique_count(),
            'mcmc_acceptances': acc_count
        }
        self.history_list.append(new_history)
        self._history_cached = False
        print(f'End of iteration, elapsed time (mins) {elapsed_time/60.:.1f}')
        print(f'Sims this iteration {self.total_sims - old_total_sims:d}')
        print(f'MCMC acceptances {acc_count:d} / {mcmc_updates:d}')

    @torch.no_grad()
    def make_mcmc_moves(self):
        """Update particles using MCMC moves

        Currently only particles with zero weight are updated.

        Returns number of MCMC updates, and number of acceptances
        """
        mcmc_updates = 0
        mcmc_acceptances = 0
        for p in self.particles.values():
            for i in range(p.count):
                mutated_state, mutated_dist, acc, nsims = self.kernel.sample(
                    p.values[[i], :], # using [i] avoids dropping first dim
                    p.distances[i]
                )
                self.total_sims += nsims
                mcmc_updates += 1
                mcmc_acceptances += acc
                p.values[i, :] = mutated_state
                p.distances[i] = mutated_dist
            p.weights = torch.ones(p.count) # Algorithm guarantees distance is below threshold, so weight is 1
        return mcmc_updates, mcmc_acceptances
