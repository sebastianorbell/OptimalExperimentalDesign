import os

import matplotlib.pyplot as plt
import numpy as np
import string
alphabet = string.ascii_lowercase

def plot_distributions(model_dict, samples=1000, bins=50, title=''):
    n = model_dict.__len__()
    fig, ax = plt.subplots(nrows=n)
    for i, (key, item) in zip(range(n), model_dict.items()):
        ax[i].hist(np.random.normal(*list(item.values()), samples), alpha=0.5, bins=bins)
        ax[i].set_title(f'{title}: {key}')
    plt.tight_layout()
    plt.show()


def plot_fit(x, y, x_pred, posteriors, fn, N=100):
    fig, ax = plt.subplots(nrows=1)
    y_pred = fn(
        x_pred.unsqueeze(-1),
        *[
            np.random.normal(
                *[
                    sub_item
                    for sub_item in item.values()
                ],
                [1, N]
            )
            for item in posteriors.values()
        ]
    )

    ymean = y_pred.mean(axis=-1)
    ystd = y_pred.std(axis=-1)
    ax.plot(x_pred, ymean, 'r')
    ax.fill_between(x_pred, ymean - 2 * ystd, ymean + 2 * ystd, color='r', alpha=0.4)
    ax.fill_between(x_pred, ymean - 4 * ystd, ymean + 4 * ystd, color='r', alpha=0.2)
    ax.scatter(x, y, marker='.')

    ax.set_ylabel('f(x)')
    ax.set_xlabel('x')
    # ax.set_ylim(-0.3, 1.2)
    plt.tight_layout()
    plt.show()


def save_plot(cd, acquisition, eig, x, y, x_pred, posteriors, fn, n, N=100):
    fig, [ax0, ax1, ax2] = plt.subplots(nrows=3)
    eig_mean = eig.mean(axis=-1)
    eig_std = eig.std(axis=-1)
    y_pred = fn(
        x_pred.unsqueeze(-1),
        *[
            np.random.normal(
                *[
                    sub_item
                    for sub_item in item.values()
                ],
                [1, N]
            )
            for item in posteriors.values()
        ]
    )

    ymean = y_pred.mean(axis=-1)
    ystd = y_pred.std(axis=-1)
    ax0.plot(x_pred, ymean, 'r')
    ax0.fill_between(x_pred, ymean - 2 * ystd, ymean + 2 * ystd, color='r', alpha=0.4)
    ax0.fill_between(x_pred, ymean - 4 * ystd, ymean + 4 * ystd, color='r', alpha=0.2)
    ax0.scatter(x, y, marker='.')

    ax0.set_ylabel('f(x)')
    ax0.set_xlabel('x')

    ax1.plot(cd, eig_mean, 'g')
    ax1.fill_between(cd.squeeze(), eig_mean - 2 * eig_std, eig_mean + 2 * eig_std, color='g', alpha=0.4)
    ax1.fill_between(cd.squeeze(), eig_mean - 4 * eig_std, eig_mean + 4 * eig_std, color='g', alpha=0.2)
    ax1.set_ylim(0, None)
    ax1.set_ylabel('Expected information gain')
    ax1.set_xlabel('x')
    ax2.plot(cd, acquisition)
    ax2.scatter(cd[np.argmax(acquisition)], np.max(acquisition), color='r', marker='*')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Acquisition fn')
    plt.tight_layout()
    plt.savefig(f'{os.getcwd()}/pyro_boed/utils/figs/{alphabet[n]}.png')
    plt.show()

def plot_eig(cd, acquisition, eig):
    fig, [ax1, ax2] = plt.subplots(nrows=2)
    eig_mean = eig.mean(axis=-1)
    eig_std = eig.std(axis=-1)
    ax1.plot(cd, eig_mean, 'g')
    ax1.fill_between(cd.squeeze(), eig_mean - 2 * eig_std, eig_mean + 2 * eig_std, color='g', alpha=0.4)
    ax1.fill_between(cd.squeeze(), eig_mean - 4 * eig_std, eig_mean + 4 * eig_std, color='g', alpha=0.2)
    ax1.set_ylim(0, None)
    ax1.set_ylabel('Expected information gain')
    ax1.set_xlabel('x')
    ax2.plot(cd, acquisition)
    ax2.scatter(cd[np.argmax(acquisition)], np.max(acquisition), color='r', marker='*')
    plt.tight_layout()
    plt.show()
