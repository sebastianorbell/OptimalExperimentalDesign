"""

Created by sebastian.orbell

"""

import pyro
import pyro.distributions as dist
import torch
from torch.distributions.constraints import positive, real

constraints = [
    real,
    positive
]


def make_marginal_guide(variables, fn, *, sigma=0.1):
    """
    Generate a marginal pyro guide for expected information gain calculations.

    :param variables: Variable prior distributions, with a dictionary structure as follows,
    e.g.
        posteriors = {
                    "a": {
                        "a_mean": torch.tensor(3.0e-1),
                        "a_std": torch.tensor(1.0e-1)
                    },
                    "b": {
                        "b_mean": torch.tensor(5.0),
                        "b_std": torch.tensor(1.0e0)
                    },
                    ...
                }
                where the function takes parametrs fn(x, *(a, b, ...)), and a_mean, a_std
                are the mean and standard deviation of a normal distribution which defines
                parameter a.
    :param fn:
        The modelled function fn(x, *(a, b, ...)), parameterised by (a, b, ...) over vector x.
    :param sigma:
        The approximate variance of the sampled data distribution.
    :return:
        A marginal pyro guide
    """

    def marginal_guide(designs, observation_labels, target_labels):
        latents = [pyro.sample(
            key,
            dist.Normal(
                *[
                    pyro.param(sub_key, sub_item, constraint=constraint)
                    for constraint, (sub_key, sub_item) in zip(constraints, item.items())
                ])
        )
            for key, item in variables.items()
        ]

        mean = fn(designs, *latents)

        with pyro.plate('data', designs.shape[0]):
            return pyro.sample('y', dist.Normal(mean, torch.tensor(sigma)))

    return marginal_guide


def make_guide(variables, fn, *, sigma=0.1):
    """
    Make a pyro guide, based on normally distributed parameters
    which are parameterised by a mean that is constrained to be
    real, and a variance constrained to be positive.

    :param variables: Variable prior distributions, with a dictionary structure as follows,
    e.g.
        priors = {
                    "a": {
                        "a_mean": torch.tensor(3.0e-1),
                        "a_std": torch.tensor(1.0e-1)
                    },
                    "b": {
                        "b_mean": torch.tensor(5.0),
                        "b_std": torch.tensor(1.0e0)
                    },
                    ...
                }
                where the function takes parametrs fn(x, *(a, b, ...)), and a_mean, a_std
                are the mean and standard deviation of a normal distribution which defines
                parameter a.
    :param fn:
        The modelled function fn(x, *(a, b, ...)), parameterised by (a, b, ...) over vector x.
    :param sigma:
        The approximate variance of the sampled data distribution.
    :return:
        A conditioned pyro guide
    """

    def guide(x, y):
        latents = [pyro.sample(
            key,
            dist.Normal(
                *[
                    pyro.param(sub_key, sub_item, constraint=constraint)
                    for constraint, (sub_key, sub_item) in zip(constraints, item.items())
                ])
        )
            for key, item in variables.items()
        ]

        mean = fn(x, *latents)

        with pyro.plate('data', x.shape[0]):
            return pyro.sample('y', dist.Normal(mean, torch.tensor(sigma)))

    return guide
