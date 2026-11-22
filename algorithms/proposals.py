import numpy as np
import torch
from nflows import distributions, flows
from graspologic.cluster import AutoGMMCluster

class Proposal:
    def __init__(self):
        """Initialise proposal"""
        raise NotImplementedError('Abstract method')
    
    def sample_and_log_prob(self, theta=None):
        """Given parameters theta, return a proposal and its log density"""
        raise NotImplementedError('Abstract method')
    
    def log_prob(self, theta, theta_old=None):
        """Return log density of proposing theta from theta_old"""
        raise NotImplementedError('Abstract method')


class ClassicRW(Proposal):
    """The classic random walk proposal"""
    @torch.no_grad()
    def __init__(self, particles, use_weights=True):
        self.particles = particles
        if use_weights:
            cov = torch.cov(particles.values.T, aweights=particles.weights)
        else:
            cov = torch.cov(particles.values.T)
        if self.particles.dim == 1:
            self.noise = torch.distributions.Normal(
                loc = torch.zeros(1),
                scale = torch.sqrt(2.*cov)
            )
        else:
            self.noise = torch.distributions.MultivariateNormal(
                loc = torch.zeros(self.particles.dim),
                covariance_matrix = 2.*cov
            )

    @torch.no_grad()
    def sample_and_log_prob(self, theta):
        inc = self.noise.sample([1,])
        log_prob = self.noise.log_prob(inc)
        return theta + inc, log_prob

    @torch.no_grad()
    def log_prob(self, theta, theta_old=None):
        if theta_old is None:
            raise ValueError('Need to provide theta_old for random walk proposal. '
                             'ClassicRW cannot be used in kernels requiring an independence proposal, '
                             'such as the independence one-hit kernel.')
        return self.noise.log_prob(theta - theta_old)


class DiagonalRW(Proposal):
    """Gaussian random walk proposal with diagonal covariance

    `sd` is a 1-D tensor of the standard deviations for each parameter
    """
    @torch.no_grad()
    def __init__(self, sd):
        sd = torch.as_tensor(sd, dtype=torch.float32)
        self.noise = torch.distributions.Independent(
            torch.distributions.Normal(loc=torch.zeros_like(sd), scale=sd),
            1
        )

    @torch.no_grad()
    def sample_and_log_prob(self, theta):
        inc = self.noise.sample([1,])
        log_prob = self.noise.log_prob(inc)
        return theta + inc, log_prob

    @torch.no_grad()
    def log_prob(self, theta, theta_old=None):
        if theta_old is None:
            raise ValueError('Need to provide theta_old for random walk proposal. '
                             'DiagonalRW cannot be used in kernels requiring an independence proposal, '
                             'such as the independence one-hit kernel.')
        return self.noise.log_prob(theta - theta_old)


class ClassicInd(Proposal):
    """The classic independence proposal (as used in PMC ABC)"""
    @torch.no_grad()
    def __init__(self, particles, use_weights=True):
        self.particles = particles
        if use_weights:
            cov = torch.cov(particles.values.T, aweights=particles.weights)
        else:
            cov = torch.cov(particles.values.T)
        if self.particles.dim == 1:
            self.noise = torch.distributions.Normal(
                loc = torch.zeros(1),
                scale = torch.sqrt(2.*cov)
            )
        else:
            self.noise = torch.distributions.MultivariateNormal(
                loc = torch.zeros(self.particles.dim),
                covariance_matrix = 2.*cov
            )

    @torch.no_grad()
    def sample_and_log_prob(self, theta=None):
        theta, _ = self.particles.sample(1)
        theta_prop = theta + self.noise.sample([1,])
        return theta_prop, self.log_prob(theta_prop, theta)

    @torch.no_grad()
    def log_prob(self, theta, theta_old=None):
        v = self.particles.values
        w = self.particles.weights
        w = w / w.sum()
        diffs = theta.unsqueeze(0) - v.unsqueeze(1) # (n_train,n_theta,d)
        densities = torch.exp(self.noise.log_prob(diffs)) # (n_train,n_theta) hopefully
        if len(densities.shape) == 3: # Can happen in d=1 case
            densities = densities.squeeze(2)
        densities = densities * w.unsqueeze(1)
        return torch.log(densities.sum(0))


class FlowProposal(Proposal):
    """A proposal based on a normalising flow"""
    def __init__(self, transform, ninputs):
        self.base_dist = distributions.StandardNormal(shape=[ninputs])
        self.flow = flows.Flow(transform, self.base_dist)

    def sample_and_log_prob(self, theta=None):
        return self.flow.sample_and_log_prob(1)

    def log_prob(self, theta, theta_old=None):
        if len(theta.shape) > 1:
            lp = self.flow.log_prob(theta)
        else:
            lp = self.flow.log_prob(theta.unsqueeze(1)).squeeze()
        return lp
        

class MixtureProposal(Proposal):
    """A proposal based on a Gaussian mixture"""
    def __init__(self, particles, min_components=5, max_components=5, safe_label_init=False, use_weights=True):
        train = particles.values.detach().numpy()
        weights = particles.weights.detach().numpy()
        if use_weights:
            train = train[weights>0]
        if safe_label_init:
            # A naive label initialisation. Should avoid problems with singleton components, but may be slower.
            # Only suitable when min_components == max_components.
            n = train.shape[0]
            label_init = np.linspace(0,n-1,n) % min_components
        else:
            # Use default behaviour - based on another clustering method. Occasionally causes crashes with warnings
            # about singleton components.
            label_init = None
        self.mix = AutoGMMCluster(
            min_components=min_components,
            max_components=max_components,
            label_init=label_init
        )
        self.mix.fit(train)

    def sample_and_log_prob(self, theta=None):
        theta_prop, _ = self.mix.model_.sample(1)
        return torch.Tensor(theta_prop), self.log_prob(theta_prop)

    def log_prob(self, theta, theta_old=None):
        if theta is torch.Tensor:
            theta = theta.detach().numpy()
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        log_prob = self.mix.model_.score_samples(theta)
        return torch.Tensor(log_prob)


class DefensiveProposal(Proposal):
    """A defensive mixture between the prior and another proposal"""
    def __init__(self, prior, main_proposal, defensive_proportion=0.05):
        self.prior = prior
        self.main_proposal = main_proposal
        self.defensive_proportion = defensive_proportion
        if not (0. < defensive_proportion < 1.):
            raise ValueError('defensive_proportion must be in (0,1) for DefensiveProposal')

    @torch.no_grad()
    def sample_and_log_prob(self, theta=None):
        u = np.random.uniform()
        if u < self.defensive_proportion:
            if isinstance(self.prior, distributions.Distribution):
                theta_prop, prior_log_prob = self.prior.sample_and_log_prob(1)
            else: # The other expected case is a torch distribution
                theta_prop = self.prior.sample([1,])
                prior_log_prob = self.prior.log_prob(theta_prop)
            prior_prob = self.defensive_proportion * torch.exp(prior_log_prob)
            main_prob = (1.-self.defensive_proportion) * torch.exp(self.main_proposal.log_prob(theta_prop))
        else:
            theta_prop, log_main_prob = self.main_proposal.sample_and_log_prob(theta)
            prior_prob = self.defensive_proportion * torch.exp(self.prior.log_prob(theta_prop))
            main_prob = (1.-self.defensive_proportion) * torch.exp(log_main_prob)
        log_prob = torch.log(prior_prob + main_prob)
        return theta_prop, log_prob

    @torch.no_grad()
    def log_prob(self, theta, theta_old=None):
        prior_prob = self.defensive_proportion * torch.exp(self.prior.log_prob(theta))
        main_prob = (1.-self.defensive_proportion) * torch.exp(self.main_proposal.log_prob(theta, theta_old))
        return torch.log(prior_prob + main_prob)