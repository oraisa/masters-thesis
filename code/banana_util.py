#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import mmd
import banana_model

class Experiment:
    def __init__(self, dim, n0, a, n):
        self.dim = dim
        self.n0 = n0
        self.a = a
        self.n = n

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("index", type=int)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    return args

def set_seed(init_seed, index):
    np.random.seed(init_seed)
    for i in range(index + 1):
        seed = np.random.randint(2**32)
    np.random.seed(seed)

def get_problem(args):
    experiments = {
        "easy_2d": Experiment(dim=2, n0=1, a=20, n=100000),
        "hard_2d": Experiment(dim=2, n0=1, a=40, n=100000),
        "easy_10d": Experiment(dim=10, n0=1, a=20, n=200000),
        "tempered_2d": Experiment(dim=2, n0=1000, a=5, n=100000),
        "tempered_10d": Experiment(dim=10, n0=1000, a=5, n=200000),
        "gauss_50d": Experiment(dim=50, n0=1, a=0, n=200000)
    }
    exp = experiments[args.experiment]
    problem = banana_model.get_problem(exp.dim, exp.n0, exp.a, exp.n)
    problem.theta0 += np.random.normal(scale=0.02, size=problem.dim)
    return problem

def save_results(
        name, args, problem, chain, accepts, clipped_r, clipped_grad, iters,
        grad_accesses, batch_size=None
):
    posterior = problem.true_posterior
    dim = problem.dim
    n, data_dim = problem.data.shape
    epsilon = args.epsilon
    delta = 0.1 / n

    if iters > 0:
        batch_divisor = n if batch_size is None else batch_size
        final_chain = chain[int((iters - 1) / 2) + 1:, :]
        acceptance = accepts / iters
        r_clipping = np.sum(clipped_r) / iters / batch_divisor
        grad_clipping = np.nan if clipped_grad is None else (clipped_grad / batch_divisor / grad_accesses)
        mean_error = mmd.mean_error(final_chain, posterior)
        cov_error = mmd.cov_error(final_chain, posterior)
        mmd_res = mmd.mmd(final_chain, posterior)
    else:
        acceptance = np.nan
        r_clipping = np.nan
        grad_clipping = np.nan
        mean_error = np.nan
        cov_error = np.nan
        mmd_res = np.nan

    result = pd.DataFrame({
        "epsilon": [epsilon],
        "delta": [delta],
        "experimen": [args.experiment],
        "dim": [dim],
        "tempering": [problem.temp_scale != 1],
        "i": [args.index],
        "algo": [name],
        "acceptance": [acceptance],
        "r_clipping": [r_clipping],
        "grad_clipping": [grad_clipping],
        "mmd": [mmd_res],
        "mean error": [mean_error],
        "cov error": [cov_error]
    })
    result.to_csv(args.output, header=False, index=False)
