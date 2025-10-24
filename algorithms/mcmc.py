import torch
import numpy as np

def makeKernel(kernel_name="naive", **kwargs):
    if kernel_name=="naive":
        return Naive(**kwargs)
    elif kernel_name=="1_hit":
        return OneHit(**kwargs)
    elif kernel_name=="ind_1_hit":
        return Independence_OneHit(**kwargs)
    elif kernel_name=="r_hit":
        return RHit(**kwargs)
    elif kernel_name=="multi_r_hit":
        return RHit(multi=True, **kwargs)
    else:
        raise ValueError("Unknown kernel name")

class Kernel:
    def __init__(
        self,
        simulator=None,
        prior=None,
        epsilon=np.inf,
        get_distance=None,
        observations=None,
        proposal=None,
        r=2
    ):
        """Initialise MCMC kernel
        
        `simulator` should be a function with parameters as arguments, returning simulated data
        `prior` should be an object with a `log_prob` method
        `epsilon` should be a float (and will often be updated durin the ABC-SMC run)
        `get_distance` should be a function with two arguments (observed and simulated data), returning a float
        `observations` is the observed data
        `proposal` should be a Proposal object
        `r` is an int of how many hits are needed for a proposal to be accepted. Not all subclasses will use this.
        """
        self.simulator = simulator
        self.prior = prior
        self.epsilon = epsilon
        self.get_distance = get_distance
        self.observations = observations
        self.proposal = proposal
        self.r = r

    def sample(self, theta, dist):
        """Make and return a MCMC update starting from state `theta` (parameters), `dist` (ABC distance)

        Returns tuple of new parameters, new distance, acceptance (True/False), number of simulator calls.
        """
        raise NotImplementedError('Abstract method')
    
class Naive(Kernel):
    @torch.no_grad()
    def sample(self, theta, dist):
        theta_prop, forward_proposal_log_density = self.proposal.sample_and_log_prob(theta)
        reverse_proposal_log_density = self.proposal.log_prob(theta, theta_prop)
        log_ratio = reverse_proposal_log_density - forward_proposal_log_density + \
                    self.prior.log_prob(theta_prop) - self.prior.log_prob(theta)
        u = torch.rand(1)
        if torch.log(u) < log_ratio:
            sim_data = self.simulator(theta_prop)
            dist_new = self.get_distance(self.observations, sim_data)
            if dist_new <= self.epsilon:
                return theta_prop, dist_new, True, 1 # Accept proposal
        return theta, dist, False, 1 # Reject proposal

class OneHit(Kernel):
    @torch.no_grad()
    def sample(self, theta, dist):
        theta_prop, forward_proposal_log_density = self.proposal.sample_and_log_prob(theta)
        reverse_proposal_log_density = self.proposal.log_prob(theta, theta_prop)
        log_ratio = reverse_proposal_log_density - forward_proposal_log_density + \
                    self.prior.log_prob(theta_prop) - self.prior.log_prob(theta)
        u = torch.rand(1)
        if torch.log(u) > log_ratio:
            return  theta, dist, False, 0 # Reject proposal
        nsims = 0
        while True:
            sim_data = self.simulator(theta_prop)
            nsims += 1
            dist_new = self.get_distance(self.observations, sim_data)
            if dist_new <= self.epsilon:
                return theta_prop, dist_new, True, nsims # Accept proposal
            sim_data = self.simulator(theta)
            nsims += 1
            dist_new = self.get_distance(self.observations, sim_data)
            if dist_new <= self.epsilon:
                return theta, dist, False, nsims # Reject proposal

class Independence_OneHit(Kernel):
    @torch.no_grad()
    def sample(self, theta, dist):
        nsims = 0
        old_log_density = self.proposal.log_prob(theta)
        while True:
            theta_prop, prop_log_density = self.proposal.sample_and_log_prob()
            sim_data = self.simulator(theta_prop)
            nsims += 1
            dist_new = self.get_distance(self.observations, sim_data)
            if dist_new <= self.epsilon:
                break
        log_ratio = old_log_density - prop_log_density + \
                    self.prior.log_prob(theta_prop) - self.prior.log_prob(theta)
        u = torch.rand(1)
        if torch.log(u) < log_ratio:
            return theta_prop, dist_new, True, nsims # Accept proposal
        else:
            return theta, dist, False, nsims # Reject proposal

class RHit(Kernel):
    def __init__(
        self,
        multi=False,
        **kwargs
    ):
        if kwargs['r']<2:
            raise ValueError("RHit kernel requires r>=2")
        self.multi = multi
        super().__init__(**kwargs)

    @torch.no_grad()
    def sample(self, theta, dist):
        sims1 = 0
        hits = 0
        to_keep = torch.randint(1, self.r, (1,)).item() # Which hit to keep: uniform sample from (1,2,...,r-1)
        if not self.multi:
            theta_prop_next, forward_proposal_log_density_next = self.proposal.sample_and_log_prob(theta)
        while hits < self.r:
            if self.multi:
                theta_prop_next, forward_proposal_log_density_next = self.proposal.sample_and_log_prob(theta)
            sim_data = self.simulator(theta_prop_next)
            sims1 += 1
            dist_next = self.get_distance(self.observations, sim_data)
            if dist_next <= self.epsilon:
                hits += 1
                if hits == to_keep:
                    theta_prop = theta_prop_next
                    dist_new = dist_next
                    forward_proposal_log_density = forward_proposal_log_density_next

        reverse_proposal_log_density = self.proposal.log_prob(theta, theta_prop)
        sims2 = 0
        hits = 0
        while hits < self.r-1:
            if self.multi:
                theta_prop_next, _ = self.proposal.sample_and_log_prob(theta_prop)
                sim_data = self.simulator(theta_prop_next)
            else:
                sim_data = self.simulator(theta)
            sims2 += 1
            dist_next = self.get_distance(self.observations, sim_data)
            if dist_next <= self.epsilon:
                hits += 1

        nsims = sims1 + sims2
        log_ratio = reverse_proposal_log_density - forward_proposal_log_density + \
                    self.prior.log_prob(theta_prop) - self.prior.log_prob(theta)
        acc_ratio = torch.exp(log_ratio) * sims2 / (sims1 - 1.)
        u = torch.rand(1)
        if u < acc_ratio:
            return theta_prop, dist_new, True, nsims # Accept proposal
        else:
            return theta, dist, False, nsims # Reject proposal
