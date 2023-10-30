"""

Created by sebastian.orbell

"""
import numpy as np
import pyro
import torch
from tqdm import tqdm

from pyro_boed.engines.inference import infer
from pyro_boed.engines.optimal_design import expected_information_gain
from pyro_boed.models.functions.lorentzian import lorentzian
from pyro_boed.models.guides import make_guide, make_marginal_guide
from pyro_boed.models.models import make_model, make_conditioned_model
from pyro_boed.utils.plotting import plot_distributions, plot_fit, plot_eig, save_plot

def optimal_experimental_design(priors, x_min, x_max, n_points, measurement, number_of_experiments=20, initial_points=4,
                                n_iters=4000, random_sample=False, plot=True):
    target_labels = priors.keys()
    observation_labels = 'y'
    ### Prepare candidate experiments
    x_grid = torch.linspace(x_min, x_max, n_points)
    candidate_designs = x_grid.clone().unsqueeze(-1).tolist()

    ### ---------------------------------------------------------------------------------
    ### Begin experiment
    ### Sample initial points
    xs = torch.tensor(np.random.uniform(x_min, x_max, initial_points)) \
        if random_sample else torch.linspace(x_min, x_max, initial_points)
    ys = measurement(xs)

    ### Condition model and guide on priors
    conditioned_model = make_conditioned_model(
        priors,
        lorentzian
    )
    guide = make_guide(
        priors,
        lorentzian
    )

    posterior_list = []
    info_gain_list = []
    for experiment in tqdm(range(number_of_experiments)):
        pyro.clear_param_store()
        ### Calculate posteriors using the measured data
        posteriors = infer(
            xs,
            ys,
            conditioned_model,
            guide,
            priors,
            num_iters=n_iters
        )

        plot_fit(xs, ys, x_grid, posteriors, lorentzian)

        ### Compute expected information gain using marginal distributions
        candidate_designs_copy = torch.tensor(candidate_designs.copy())
        model = make_model(
            posteriors,
            lorentzian
        )
        marginal_guide = make_marginal_guide(
            posteriors,
            lorentzian
        )
        acquisition_fn, eig_samples = expected_information_gain(
            candidate_designs_copy,
            model,
            marginal_guide,
            observation_labels,
            target_labels
        )
        x_new = torch.tensor(candidate_designs.pop(np.argmax(acquisition_fn)))
        plot_eig(candidate_designs_copy, acquisition_fn, eig_samples)
        y_new = measurement(x_new)
        xs = torch.cat([xs, x_new])
        ys = torch.cat([ys, y_new])
        posterior_list.append(posteriors)
        info_gain_list.append(acquisition_fn.max())

    # save_plot(candidate_designs_copy, acquisition_fn, eig_samples, xs, ys, x_grid, posteriors, lorentzian, experiment)

    ### Calculate posteriors using the measured data
    posteriors = infer(
        xs,
        ys,
        conditioned_model,
        guide,
        priors,
        num_iters=n_iters
    )

    if plot:
        plot_distributions(posteriors, title="Posteriors")
        plot_fit(xs, ys, x_grid, posteriors, lorentzian)

    return xs, ys, posteriors, posterior_list, info_gain_list
