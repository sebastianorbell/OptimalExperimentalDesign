"""

Created by sebastian.orbell

"""

import pyro
import pyro.distributions as dist
import torch


def make_model(variables, fn, *, sigma=0.1):
    """
    Generate a pyro model for expected information gain calculations.

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
        A pyro model
    """

    def model(designs):
        latents = [pyro.sample(key, dist.Normal(*[sub_item
                                                  for _, sub_item in item.items()]))
                   for key, item in variables.items()]
        mean = fn(designs, *latents)
        with pyro.plate('data', designs.shape[0]):
            return pyro.sample('y', dist.Normal(mean, torch.tensor(sigma)))

    return model


def make_conditioned_model(variables, fn, *, sigma=0.1):
    """
    Generate a conditioned pyro model for variational inferenc.

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
        A conditioned pyro model
    """

    def conditioned_model(x, y):
        latents = [pyro.sample(key, dist.Normal(*[sub_item
                                                  for _, sub_item in item.items()]))
                   for key, item in variables.items()]
        mean = fn(x, *latents)
        with pyro.plate('data', x.shape[0]):
            return pyro.sample('y', dist.Normal(mean, torch.tensor(sigma)), obs=y)

    return conditioned_model
