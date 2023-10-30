"""

Created by sebastian.orbell

"""

import pyro
from pyro.optim import AdagradRMSProp, DCTAdam
from pyro.infer import SVI, Trace_ELBO
from ..utils.utils import deep_copy_dict

def infer(x,
          y,
          conditioned_model,
          guide,
          priors, *,
          num_iters=5000):
    """
    Stochastic Variational Inference to estimate the
    distributions of parameters in a model, conditioned
    upon a set of observations (xs, ys).

    :param x: Vector of sampled coordinates.
    :param y: Measurement values at sampled coordinates.
    :param conditioned_model: Conditioned pyro model
    :param guide: Conditioned pyro guide
    :param priors: Prior distributions
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
                where the function, which the guide is conditioned upon, takes parametrs
                fn(x, *(a, b, ...)), and a_mean, a_std are the mean and standard deviation
                of a normal distribution which defines parameter a.
    :param num_iters:
        The number of optimisation iterations for variational inference.
    :return:
        posteriors - the porsterior distributions, defined in the same nested dictionary
        structure as the prior distributions.
    """
    posteriors = deep_copy_dict(priors)

    svi = SVI(conditioned_model,
              guide,
              AdagradRMSProp({}),
              loss=Trace_ELBO())

    for _ in range(num_iters):
        elbo = svi.step(x, y)

    for key, item in posteriors.items():
        for sub_key, sub_item in item.items():
            posteriors[key][sub_key] = pyro.param(sub_key).detach().clone()

    return posteriors
