import numpy as np
from dp_mcmc_module.barker_mog import run_dp_Barker
from dp_mcmc_module.exact_rdp import get_privacy_spent
from dp_mcmc_module.X_corr import load_x_corr
import pickle
import util

class BarkerParams:
    def __init__(self, batch_size, prop_sigma):
        self.batch_size = batch_size
        self.prop_sigma = prop_sigma

def rdp_epsilon(iters, delta, n, b):
    return get_privacy_spent(
        b, n, iters, max_alpha=int(b / 5), delta=delta
    )[0]

def maximize_iters(epsilon, delta, n, batch_size):
    low_iters = 0
    up_iters = 1024
    while rdp_epsilon(up_iters, delta, n, batch_size) < epsilon:
        up_iters *= 2
    while int(up_iters) - int(low_iters) > 1:
        new_iters = (low_iters + up_iters) / 2
        new_epsilon = rdp_epsilon(new_iters, delta, n, batch_size)
        if new_epsilon > epsilon:
            up_iters = new_iters
        else:
            low_iters = new_iters

    if rdp_epsilon(int(up_iters), delta, n, batch_size) < epsilon:
        return int(up_iters)
    else:
        return int(low_iters)


def dp_barker(problem, theta0, epsilon, delta, params, verbose=True):

    data = problem.data
    n, data_dim = data.shape
    T = maximize_iters(epsilon, delta, n, params.batch_size)
    if verbose:
        print("Iterations: {}".format(T))

    xcorr_params, n_points = load_x_corr()
    chain, clip_count, accepts = run_dp_Barker(
        problem, T, params.prop_sigma**2, theta0, problem.temp_scale,
        xcorr_params, n_points, batch_size=params.batch_size, verbose=verbose
    )
    return util.MCMCResult(
        problem, chain, chain, T, accepts,
        np.sum(clip_count) / T / params.batch_size, np.nan
    )
