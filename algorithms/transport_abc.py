import numpy as np
from copy import deepcopy
from time import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nflows
from nflows.transforms import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    ReversePermutation,
    CompositeTransform
)
from nflows.nn.nets import ResidualNet
from torch.nn.functional import softplus
from algorithms.abc_smc import ABC_SMC
# nflows 0.14's AffineCouplingTransform has no scale_activation argument, so use our version
from algorithms.nflowsutils import AffineCouplingTransform
from algorithms.utils import effective_sample_size
from algorithms.proposals import ClassicInd, DefensiveProposal, FlowProposal

class TransportABC(ABC_SMC):
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
    `transform_name` is what transform to use in the normalising flow. Options are 'spline', 'spline_two_layers', 'coupling'.
    `reinit_flow` is whether to reinitialise the flow in each training iteration
    `enable_fallback` is whether to use classic ABC when it has a better test loss
    `batch_size` is training batch size
    `kernel_name` indicates which MCMC kernel to use
    `train_target` defines the target distribution when training the flow - options are 'current' and 'previous'
    `defensive_proportion` sets up a defensive proposal if given a positive value
    """
    def __init__(
            self, N_train, N_test, simulator, n_inputs, observations, prior,
            epsilon_reduction_parameter=0.5,
            get_distance = lambda y1, y2: torch.norm(y1-y2, 2, dim=1),
            r=2,
            transform_name = 'spline',
            reinit_flow = False,
            enable_fallback = True,
            batch_size = 100,
            kernel_name = 'ind_1_hit',
            train_target = 'current',
            defensive_proportion = 0.
    ):
        self.init_common(
            simulator, n_inputs, observations, prior, "flow", # proposal must be flow
            epsilon_reduction_parameter, get_distance, r, kernel_name, train_target, defensive_proportion
        )

        print('Initialising particles')
        self.particles = {
            'train':self.init_particles(N_train),
            'test':self.init_particles(N_test)
        }
        self.batch_size = batch_size

        self.transform_name = transform_name
        self.reinit_flow = reinit_flow
        self.enable_fallback = enable_fallback
        self.nr_fallback = 0

    def init_transform(self):
        if self.transform_name == 'spline':
            return MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features = self.n_inputs,
                hidden_features=20,
                num_bins=50,
                tails='linear',
                tail_bound=10.,
                num_blocks=10
            )
        elif self.transform_name == 'spline_two_layers':
            transform_list = []
            num_layers = 2
            for i in range(num_layers):
                if i > 0:
                    transform_list.append(
                       ReversePermutation(features=self.n_inputs)
                    )
                transform_list.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        features = self.n_inputs,
                        hidden_features=20,
                        num_bins=50,
                        tails='linear',
                        tail_bound=10.,
                        num_blocks=10
                    )
                )
            return CompositeTransform(transform_list)
        elif self.transform_name == 'coupling':
            if self.n_inputs < 2:
                raise ValueError('Coupling transform requires at least 2 inputs')
            mask = torch.zeros(self.n_inputs)
            mask[self.n_inputs//2:] = 1 # First half (rounding down) of mask is 1s
            def create_net(in_features, out_features):
                return ResidualNet(
                    in_features, out_features, hidden_features=20, num_blocks=10
                )
            transform_list = []
            num_layers = 4
            for _ in range(num_layers):
                mask = 1 - mask
                tf = AffineCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=create_net,
                    scale_activation=lambda x : (softplus(x) + 1e-3).clamp(0, 3)
                )
                transform_list.append(tf)
            return CompositeTransform(transform_list)
        else:
            raise ValueError('Unknown transform name')

    def pretrain(self, check_ess_every=10, ess_target_fraction=0.9):
        """Train a transform targetting the prior"""
        with self.training_timer():
            transform = self.init_transform()
            self.flow_proposal = FlowProposal(transform, self.n_inputs)
            optimizer = Adam(self.flow_proposal.flow.parameters(), lr=1e-4)
            ess = 0.
            while ess < self.batch_size * ess_target_fraction:
                for i in range(check_ess_every):
                    if isinstance(self.prior, nflows.distributions.Distribution):
                        batch = self.prior.sample(self.batch_size)
                    else: # The other expected case is a torch distribution
                        batch = self.prior.sample([self.batch_size,])
                    loss = -self.flow_proposal.log_prob(batch).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                params = self.flow_proposal.flow.sample(self.batch_size).detach()
                logw = self.prior.log_prob(params).squeeze()
                # squeeze on prev line ensures desired shape in one-dimensional parameters case
                logw -= self.flow_proposal.log_prob(params)
                ess = effective_sample_size(logw, log_input=True)
                print(f'Pretraining ESS is {ess:.1f}', end='\r')
            print('\n', end='')
        return None

    def loss(self, proposal, pars, weights):
        """Loss (weighted cross entropy) under `proposal` for given parameters and weights"""
        loss = -proposal.log_prob(pars)
        if self.train_target == 'current':
            # For current target apply weights (i.e. only use particles with epsilon below threshold)
            loss = loss * weights
        return loss.mean() # Mean not sum so different sized inputs comparable

    def training_iteration(self):
        print(f'\nStarting ABC-SMC iteration {self.iteration+1:d}')
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
            flow_test_loss = self.learn_transform()
        self.kernel.proposal = self.flow_proposal
        self.kernel.epsilon = self.epsilon
        if self.enable_fallback:
            with self.training_timer():
                self.classic_proposal = ClassicInd(
                    self.particles['train'],
                    use_weights = (self.train_target == 'current')
                )
                classic_test_loss = self.loss(
                    self.classic_proposal,
                    self.particles['test'].values,
                    self.particles['test'].weights,
                )
            if flow_test_loss > classic_test_loss:
                print('Falling back to classic ABC proposal!')
                self.nr_fallback +=1
                self.kernel.proposal = self.classic_proposal
        if self.defensive_proportion > 0:
            self.kernel.proposal = DefensiveProposal(
                self.prior,
                self.kernel.proposal,
                defensive_proportion = self.defensive_proportion
            )

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

    def learn_transform(self):
        """Train next transform"""
        if self.reinit_flow:
            transform = self.init_transform()
            self.flow_proposal = FlowProposal(transform, self.n_inputs)
        optimizer = torch.optim.Adam(self.flow_proposal.flow.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(
            optimizer, 'min',
            patience=50
        )
        # Create training batches whose sizes are roughly equal and close to self.batch_size
        # TODO: Only base training/test sets on alive particles?
        nbatches = round(self.particles['train'].count / self.batch_size)
        train_set = self.particles['train'].values.detach()
        train_weights = self.particles['train'].weights.detach()
        train_batches = torch.tensor_split(train_set, nbatches)
        weights_batches = torch.tensor_split(train_weights, nbatches)
        best_test_loss = np.inf
        best_pars = self.flow_proposal.flow.state_dict()
        best_pars = deepcopy(best_pars)
        epochs_since_best = 0
        iterations = 0
        while epochs_since_best < 100: #TODO: replace threshold with parameter?
            train_loss = 0.
            for batch_counter in range(nbatches):
                optimizer.zero_grad()
                loss = self.loss(
                    self.flow_proposal,
                    train_batches[batch_counter],
                    weights_batches[batch_counter]
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= nbatches
            with torch.no_grad():
                test_loss = self.loss(
                    self.flow_proposal,
                    self.particles['test'].values,
                    self.particles['test'].weights
                ).item()
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]['lr']
            iterations += 1
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_pars = self.flow_proposal.flow.state_dict()
                best_pars = deepcopy(best_pars)
                epochs_since_best = 0
            else:
                epochs_since_best += 1   
            print(
                f'Train steps {iterations}, train loss {train_loss:.4e}, '
                f'test loss {test_loss:.4e}, best test loss {best_test_loss:.4e}, '
                f'learning rate {current_lr:.1e}, '
                f'epochs since best {epochs_since_best}   ',
                end='\r'
            )
        print('\n', end='')
        self.flow_proposal.flow.load_state_dict(best_pars)
        return best_test_loss
