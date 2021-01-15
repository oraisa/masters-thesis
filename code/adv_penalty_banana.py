#!/usr/bin/env python3

import numpy as np
import dp_penalty
import banana_util

args = banana_util.parse_args()
banana_util.set_seed(46327, args.index)

problem = banana_util.get_problem(args)
n, data_dim = problem.data.shape

epsilon = args.epsilon
delta = 0.1 / n

param_dict = {
    "easy_2d": dp_penalty.PenaltyParams(
        tau = 0.1,
        prop_sigma = np.array((0.008, 0.007)) * 2,
        r_clip_bound = 1.8,
        ocu = True,
        grw = True
    ),
    "hard_2d": dp_penalty.PenaltyParams(
        tau = 0.1,
        prop_sigma = np.array((0.008, 0.007)) * 2,
        r_clip_bound = 1.8,
        ocu = True,
        grw = True
    ),
    "easy_10d": dp_penalty.PenaltyParams(
        tau = 0.08,
        prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.02,
        r_clip_bound = 3,
        ocu = True,
        grw = True
    ),
    "tempered_2d": dp_penalty.PenaltyParams(
        tau = 0.1,
        prop_sigma = np.array((0.008, 0.007)) * 20,
        r_clip_bound = 3,
        ocu = True,
        grw = True
    ),
    "tempered_10d": dp_penalty.PenaltyParams(
        tau = 0.08,
        prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.0005,
        r_clip_bound = 3,
        ocu = True,
        grw = True
    ),
    "gauss_50d": dp_penalty.PenaltyParams(
        tau = 0.08,
        prop_sigma = np.hstack((np.array((8, 7)), np.repeat(5, 48))) * 0.02,
        r_clip_bound = 3,
        ocu = True,
        grw = True
    )
}
params = param_dict[args.experiment]

chain, accepts, clipped_r, iters = dp_penalty.dp_penalty(problem, epsilon, delta, params)
banana_util.save_results("dppa", args, problem, chain, accepts, clipped_r, None, iters, None)
