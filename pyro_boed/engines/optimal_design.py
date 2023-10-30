"""

Created by sebastian.orbell

"""

import pyro
from pyro.optim import AdagradRMSProp, DCTAdam
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.oed.eig import marginal_eig
from scipy.stats import norm
import numpy as np


def acquisition(y, xi=1e-8):
    """
    Returns the aquisition function based upon
    the expected information gain. This
    acquisition function calculates the
    expected improvement.
    :param y: The expected information gain
    :param xi: factor for numerical stability
    :return: Acquisition data
    """
    y_mean = y.mean(axis=-1)
    y_std = y.std(axis=-1)
    y_opt = y_mean.max()

    with np.errstate(divide='warn'):
        imp = y_mean - y_opt - xi
        Z = imp / y_std
        ei = imp * norm.cdf(Z) + y_std * norm.pdf(Z)
        ei[y_std == 0.0] = 0.0

    return ei


def expected_information_gain(designs,
                              model,
                              marginal_guide,
                              observation_labels,
                              target_labels,
                              *,
                              num_steps=200,
                              num_samples=100,
                              final_num_samples=400):
    """
    Calculates the expected information gain
    over a set of designs and based upon a
    model of the system. The Bayesian model
    is conditioned upon a set of observations.

    :param designs: Set of experimental designs,
        coordinates over a vector space
    :param model: Numerical model of the system
    :param marginal_guide: pyro marginal guide
    :param observation_labels: A subset of the sample sites
        present in `model`. These sites are regarded as future observations
        and other sites are regarded as latent variables over which a
        posterior is to be inferred.
    :param list target_labels: A subset of the sample sites over which the posterior
        entropy is to be measured.
    :param int num_samples: Number of samples per iteration.
    :param int num_steps: Number of optimization steps.
    :param final_num_samples:
    :return:
        acq_fn: The acquisition function evaluated over the design
            set.
        eig_samples: The expected information gain sample populations
            over the design set.
    """
    eig = marginal_eig(
        model,
        designs,
        observation_labels,
        target_labels,
        num_samples=num_samples,
        num_steps=num_steps,
        guide=marginal_guide,
        optim=AdagradRMSProp({}),
        final_num_samples=final_num_samples
    )
    eig_samples = eig.detach().numpy()
    acq_fn = acquisition(eig_samples)

    return acq_fn, eig_samples
