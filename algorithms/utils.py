import os
import tempfile
from os.path import exists
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scipy.stats import norm
import ot

def wdist(abc, ref):
    """Wasserstein distance between two numpy arrays"""
    ot_mat = ot.dist(abc, ref, metric='euclidean')
    a = ot.unif(len(abc))
    b = ot.unif(len(ref))
    return ot.emd2(a, b, ot_mat)

def wdist_se(abc, ref, n=100):
    """Bootstrap standard error of Wasserstein distance between two numpy arrays

    The ABC sample is bootstrapped but not the reference sample"""
    nabc = abc.shape[0]
    bootstrap_samples = [
        wdist(abc[np.random.choice(nabc,nabc),...], ref)
        for i in range(n)
    ]
    return np.std(np.array(bootstrap_samples))


standard_normal = torch.distributions.Normal(0., 1.)

def norm_to_unif(x, a=0., b=1.):
    """Convert N(0,1) draws to U(a,b) draws"""
    if isinstance(x, torch.Tensor):
        return a + (b-a)*standard_normal.cdf(x)
    else:
        return a + (b-a)*norm.cdf(x)
    
def unif_to_norm(x, a=0., b=1.):
    """Convert U(a,b) draws to N(0,1) draws"""
    if isinstance(x, torch.Tensor):
        return standard_normal.icdf((x-a)/(b-a))
    else:
        return norm.ppf((x-a)/(b-a))

def effective_sample_size(w, log_input=False):
    """Effective sample size of weights or log_weights

    `w` is a 1-D tensor of weights or log weights (can be unnormalised in either case)
    `log_input` denotes whether `w` is log weights
    """
    if log_input:
        log_weights = w
        max_log_weight = log_weights.max()
        if max_log_weight > -np.inf:
            weights = torch.exp(log_weights - max_log_weight)
        else:
            return torch.tensor(0.)
    else:
        weights = w

    sum_weights = weights.sum()
    if sum_weights == 0:
        return torch.tensor(0.)

    ess = (sum_weights ** 2.0) / (weights ** 2.0).sum()
    return ess

class Particles:
    """Weighted particles

    `count` - number of particles
    `values` - matrix whose rows give particle values
    `weights` - vector of particle weights
    `distances` - vector of ABC distances for each particle

    (nb `values`, `weights` and `distances` are all tensors)
    """
    def __init__(self, count, values, weights, distances):
        if len(values.shape) == 1:
            values.unsqueeze_(1)
        assert(len(values.shape) == 2)
        assert(len(weights.shape) == 1)
        assert(len(distances.shape) == 1)
        assert(values.shape[0] == count)
        assert(weights.shape[0] == count)
        assert(distances.shape[0] == count)
        self.count = count
        self.dim = values.shape[1]
        self.values = values
        self.weights = weights
        self.distances = distances

    @torch.no_grad()
    def sample(self, n, method='multinomial', u=None):
        """Return `n` sampled values

        A tuple of sampled values and distances is returned

        The default method is `multinomial`, and `systematic` is also supported.

        `u` is a single uniform random sample to be used in systematic sampling.
        If not supplied this will be generated here.
        """
        if method == "multinomial":
            i = torch.multinomial(self.weights, n, replacement=True)
        else:
            # Systematic resampling
            i = torch.zeros(n, dtype=torch.long)
            if u is None:
                u = torch.rand(1)
            v = (u + torch.arange(n)) / n
            w = self.weights.clone()
            w = w / sum(w)
            cusum = torch.cumsum(w, dim=0)
            new_count, old_count = 0, 0
            while new_count < n:
                if v[new_count] <= cusum[old_count] or old_count == n-1: # 2nd condition catches rounding errors
                    i[new_count] = old_count
                    new_count += 1
                else:
                    old_count += 1

        return self.values[i, :], self.distances[i]

    @torch.no_grad()
    def resample(self, method='multinomial', u=None):
        """Replace particles with `n` sampled values

        Details as in `sample`
        """
        new_values, new_distances = self.sample(self.count, method, u)
        self.values = new_values
        self.distances = new_distances
        self.weights = torch.ones_like(self.distances)

    def reweight(self, epsilon):
        """Set weight to zero where distance above eps"""
        self.weights = self.weights*(self.distances <= epsilon).to(torch.float32)

    def unique_count(self, eps=None):
        """Number of unique particles (at specified epsilon value if one supplied)"""
        if eps == None:
            alive = (self.weights > 0.)
        else:
            alive = (self.distances <= eps)
        if alive.sum() == 0:
            return 0
        return torch.unique(self.values[alive,:], dim=0).shape[0]

def save_summary(summary, results_dir=None):
    """Save `summary` (a dict of results from one run) as a row of a csv file

    If `results_dir` is None, the row is appended to results/results.csv.
    Otherwise it is saved to a new uniquely named file in `results_dir`,
    so that runs in parallel don't write to the same file.
    """
    df = pd.DataFrame.from_dict([summary]) # See https://stackoverflow.com/a/63972829
    if results_dir is None:
        Path('results').mkdir(exist_ok=True)
        output_path='results/results.csv'
        df.to_csv(output_path, mode='a', header=not exists(output_path))
    else:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        prefix = f"{summary['model']}_{summary['proposal']}_{summary['kernel']}_seed{summary['seed']}_"
        fd, output_path = tempfile.mkstemp(suffix='.csv', prefix=prefix, dir=results_dir)
        with os.fdopen(fd, 'w') as f:
            df.to_csv(f, index=False)
