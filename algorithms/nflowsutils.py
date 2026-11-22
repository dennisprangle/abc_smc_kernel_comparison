#This file contains extensions and variations of code from the nflows package
import torch
import numpy as np
from torch.distributions import Independent
from torch.nn.functional import softplus
from nflows.utils import torchutils
from nflows.transforms import Transform
from nflows.transforms import AffineCouplingTransform as NflowsAffineCouplingTransform
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

# Next class matches AffineCouplingTransform in the nflows repo (as of commit 30a557c, Dec 2021),
# which adds the `scale_activation` argument.
# The nflows 0.14 package (the latest release on PyPI) doesn't have this argument,
# and we can't install the repo version due to version clashes with other packages.
# So this subclasses the package version and overrides `_scale_and_shift` as in the repo version.
# One difference from the repo: `scale_activation` is set after calling the parent constructor,
# so this class also works with the repo version, whose constructor would otherwise reset it to the default.
class AffineCouplingTransform(NflowsAffineCouplingTransform):
    """An affine coupling layer that scales and shifts part of the variables.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.

    The user should supply `scale_activation`, the final activation function in the neural network producing the scale tensor.
    Two options are predefined in the class.
    `DEFAULT_SCALE_ACTIVATION` preserves backwards compatibility but only produces scales <= 1.001.
    `GENERAL_SCALE_ACTIVATION` produces scales <= 3, which is more useful in general applications.
    """

    DEFAULT_SCALE_ACTIVATION = lambda x : torch.sigmoid(x + 2) + 1e-3
    GENERAL_SCALE_ACTIVATION = lambda x : (softplus(x) + 1e-3).clamp(0, 3)

    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None, scale_activation=DEFAULT_SCALE_ACTIVATION):
        super().__init__(mask, transform_net_create_fn, unconditional_transform)
        self.scale_activation = scale_activation

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features:, ...]
        shift = transform_params[:, : self.num_transform_features, ...]
        scale = self.scale_activation(unconstrained_scale)
        return scale, shift

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