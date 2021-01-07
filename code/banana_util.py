#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import mmd
import banana_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("epsilon", type=float)
    parser.add_argument("dim", type=int)
    parser.add_argument("tempering", type=bool)
    parser.add_argument("index", type=int)
    parser.add_argument("output", type=str)
    return parser.parse_args()

def set_seed(init_seed, index):
    np.random.seed(init_seed)
    for i in range(index + 1):
        seed = np.random.randint(2**32)
    np.random.seed(seed)

def get_problem(args):
    problem = banana_model.get_problem(args.dim, args.tempering)
    problem.theta0 += np.random.normal(scale=0.02, size=args.dim)
    return problem

def save_results(name, args, problem, chain, accepts, clipped_r, clipped_grad, iters, grad_accesses):
    posterior = problem.true_posterior
    dim = args.dim
    n, data_dim = problem.data.shape
    epsilon = args.epsilon
    delta = 0.1 / n

    if iters > 0:
        final_chain = chain[int((iters - 1) / 2) + 1:, :]
        acceptance = accepts / iters
        r_clipping = np.sum(clipped_r) / iters / n
        grad_clipping = np.nan if clipped_grad is None else clipped_grad / n / grad_accesses
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
        "dim": [dim],
        "i": [args.index],
        "algo": [name],
        "acceptance": [acceptance],
        "r_clipping": [r_clipping],
        "grad_clipping": [grad_clipping],
        "mmd": [mmd_res],
        "mean error": [mean_error],
        "cov error": [cov_error]
    })
    result.to_csv(args.output, header=False)
