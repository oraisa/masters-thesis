#!/usr/bin/env python3

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

def compute_iters_dp_mcmc(eps, target_delta, N, batch_size):
    cur_eps = 0
    cur_T = 0
    increment = 1000
    while(increment > 1):
        increment = int(increment / 2)
        while(cur_eps <= eps):
            cur_T += increment
            cur_eps, _ = get_privacy_spent(
                batch_size, N, cur_T, max_alpha=int(batch_size / 5),
                delta=target_delta
            )
        cur_eps = 0
        cur_T -= increment

    return cur_T

def dp_barker(problem, epsilon, delta, params, verbose=True):

    data = problem.data
    n, data_dim = data.shape
    T = compute_iters_dp_mcmc(epsilon, delta, n, params.batch_size)
    if verbose:
        print("Iterations: {}".format(T))

    xcorr_params, n_points = load_x_corr()
    chain, clip_count, accepts = run_dp_Barker(
        problem, T, params.prop_sigma**2, problem.theta0, problem.temp_scale,
        xcorr_params, n_points, batch_size=params.batch_size, verbose=verbose
    )
    return util.MCMCResult(
        problem, chain, chain, T, accepts,
        np.sum(clip_count) / T / params.batch_size, np.nan
    )
