#!/usr/bin/env python3

import numpy as np
import banana_util
import hmc

args = banana_util.parse_args()
banana_util.set_seed(23479, args.index)

dim = args.dim
problem = banana_util.get_problem(args)
n, data_dim = problem.data.shape

epsilon = args.epsilon
delta = 0.1 / n

if args.tempering:
    if dim == 2:
        params = hmc.HMCParams(
            tau = 0.1,
            tau_g = 0.4,
            L = 10,
            eta = 0.01,
            mass = np.array((0.3, 1)),
            r_clip = 2,
            grad_clip = 1.0,
        )
    elif dim == 10:
        params = hmc.HMCParams(
            tau = 0.05,
            tau_g = 0.2,
            L = 10,
            eta = 0.01,
            mass = np.array((0.3, 1, 2, 2, 2, 2, 2, 2, 2, 2)),
            r_clip = 3,
            grad_clip = 3.0,
        )
else:
    if dim == 2:
        params = hmc.HMCParams(
            tau = 0.1,
            tau_g = 0.4,
            L = 10,
            eta = 0.0005,
            mass = np.array((0.3, 1)),
            r_clip = 2,
            grad_clip = 1.0,
        )
    elif dim == 10:
        params = hmc.HMCParams(
            tau = 0.05,
            tau_g = 0.2,
            L = 10,
            eta = 0.0002,
            mass = np.array((0.3, 1, 2, 2, 2, 2, 2, 2, 2, 2)),
            r_clip = 3,
            grad_clip = 3.0,
        )


chain, leapfrog_chain, accepts, clipped_r, iters, clipped_grad, grad_accesses = hmc.hmc(
    problem, epsilon, delta, params
)
banana_util.save_results("hmc", args, problem, chain, accepts, clipped_r, clipped_grad, iters, grad_accesses)
