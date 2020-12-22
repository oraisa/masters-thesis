#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import mcmc_animation
import banana_model
import hmc
import util
import mmd

parser = argparse.ArgumentParser()
parser.add_argument("epsilon", type=float)
parser.add_argument("dim", type=int)
parser.add_argument("index", type=int)
parser.add_argument("output", type=str)
args = parser.parse_args()

np.random.seed(46237)
for i in range(args.index + 1):
    seed = np.random.randint(2**32)
np.random.seed(seed)

dim = args.dim
banana = banana_model.BananaModel(dim, a=20)
data = banana.generate_test_data()
problem = util.Problem(banana.log_likelihood_per_sample, banana.log_prior, data)
n, data_dim = problem.data.shape

epsilon = args.epsilon
delta = 0.1 / n

params = hmc.HMCParams(
    tau = 0.1,
    tau_g = 0.4,
    L = 10,
    eta = 0.0005,
    mass = np.array((0.3, 1)),
    r_clip = 2,
    grad_clip = 1.0,
    theta0 = np.array((0.0, 3.0))
)
params.theta0 += np.random.normal(scale=0.02, size=dim)

chain, leapfrog_chain, accepts, clipped_r, iters, clipped_grad, grad_accesses = hmc.hmc(
    problem, epsilon, delta, params
)

final_chain = chain[int((iters - 1) / 2) + 1:, :]
posterior = banana.generate_posterior_samples(1000, data, 1)

acceptance = accepts / iters
r_clipping = np.sum(clipped_r) / iters / n
grad_clipping = clipped_grad / n / grad_accesses
mean_error = mmd.mean_error(final_chain, posterior)
cov_error = mmd.cov_error(final_chain, posterior)
mmd = mmd.mmd(final_chain, posterior)

result = pd.DataFrame({
    "epsilon": [epsilon],
    "delta": [delta],
    "dim": [dim],
    "i": [args.index],
    "algo": ["HMC"],
    "acceptance": [acceptance],
    "r_clipping": [r_clipping],
    "grad_clipping": [grad_clipping],
    "mmd": [mmd],
    "mean error": [mean_error],
    "cov error": [cov_error]
})
result.to_csv(args.output, header=False)
