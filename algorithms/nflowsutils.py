#This file contains extensions and variations of code from the nflows package
import torch
import numpy as np
from torch.distributions import Independent
from nflows.utils import torchutils
from nflows.transforms import Transform
from nflows.transforms.base import InputOutsideDomain

# Next class is from nflows repo, but not in my version of package
class Exp(Transform):
    def forward(self, inputs, context=None):
        outputs = torch.exp(inputs)
        logabsdet = torchutils.sum_except_batch(inputs, num_batch_dims=1)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) <= 0.:
            raise InputOutsideDomain()

        outputs = torch.log(inputs)
        logabsdet = -torchutils.sum_except_batch(outputs, num_batch_dims=1)

        return outputs, logabsdet

# Based on BoxUniform from nflows package, with some fixes (ignore context, log_prob works with inputs outside support)
# TODO: Delete if no longer used
class IndependentGamma(Independent):
    def __init__(
        self, concentration, rate, reinterpreted_batch_ndims = 1
    ):
        gamma = torch.distributions.Gamma(
            concentration, rate, validate_args=False
        )
        super().__init__(
            gamma, reinterpreted_batch_ndims
        )

    def sample(self, num_samples, context=None):
        return super().sample(num_samples)

    def log_prob(self, inputs, context=None):
        lp_no_validation = super().log_prob(inputs)
        # Correct prev line for giving wrong results outside support
        return torch.where(
            torch.all(inputs > 0., dim=1),
            lp_no_validation,
            torch.full_like(lp_no_validation, -np.inf)
        )