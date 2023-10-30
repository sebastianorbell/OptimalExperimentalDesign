"""

Created by sebastian.orbell

"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from pyro_boed.models.functions.lorentzian import lorentzian
from pyro_boed.utils.plotting import plot_distributions

### Set up test data functions
a = torch.normal(2.0e-1, 1.0e-1, (1,))
b = torch.normal(5., 1e-1, (1,))
# s = torch.normal(1.0e-1, 1.0e-3, (1,))
s = 0.1
print(f'a = {a}, b = {b}')

x_min = 0
x_max = 10

# Mock measurement function
measurement = lambda x: torch.normal(lorentzian(x, a, b), s)

### Set up priors
priors = {
    "a": {
        "a_mean": torch.tensor(3.0e-1),
        "a_std": torch.tensor(1.0e-1)
    },
    "b": {
        "b_mean": torch.tensor(5.0),
        "b_std": torch.tensor(1.0e0)
    },
}

plot_distributions(priors, title="Priors")
from pyro_boed.loop import optimal_experimental_design

n_points = 200
xs, ys, posteriors, posterior_list, info_gain_list = optimal_experimental_design(priors, x_min, x_max, n_points,
                                                                                 measurement, number_of_experiments=20,
                                                                                 initial_points=10, n_iters=5000,
                                                                                 random_sample=True, plot=True)

a_mean = np.array([p['a']['a_mean'] for p in posterior_list])
a_std = np.array([p['a']['a_std'] for p in posterior_list])
b_mean = np.array([p['b']['b_mean'] for p in posterior_list])
b_std = np.array([p['b']['b_std'] for p in posterior_list])
n = np.arange(a_mean.__len__())

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(n, a_mean, '--', color='k')
axes[0].fill_between(n, a_mean - 2 * a_std, a_mean + 2 * a_std, alpha=0.5, color='orange')
axes[0].set_ylabel('Parameter a')

axes[1].plot(n, b_mean, '--', color='k')
axes[1].fill_between(n, b_mean - 2 * b_std, b_mean + 2 * b_std, alpha=0.5, color='orange')

axes[1].set_ylabel('Parameter b')
axes[1].set_xlabel('Number of experiments')
plt.show()
