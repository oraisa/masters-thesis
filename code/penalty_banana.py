#!/usr/bin/env python3

import numpy as np
import banana_util
import dp_penalty

args = banana_util.parse_args()
banana_util.set_seed(23648, args.index)

problem = banana_util.get_problem(args)
n, data_dim = problem.data.shape

epsilon = args.epsilon
delta = 0.1 / n

param_dict = {
    "easy_2d": dp_penalty.PenaltyParams(
        tau = 0.1,
        prop_sigma = np.array((0.008, 0.007)),
        r_clip_bound = 2,
        ocu = False,
        grw = False
    ),
    "hard_2d": dp_penalty.PenaltyParams(
        tau = 0.1,
        prop_sigma = np.array((0.008, 0.007)),
        r_clip_bound = 2,
        ocu = False,
        grw = False
    ),
    "easy_10d": dp_penalty.PenaltyParams(
        tau = 0.08,
        prop_sigma = np.array((0.008, 0.007, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001)) * 0.8,
        r_clip_bound = 2,
        ocu = False,
        grw = False
    ),
    "tempered_2d": dp_penalty.PenaltyParams(
        tau = 0.1,
        prop_sigma = np.array((0.008, 0.007)) * 12,
        r_clip_bound = 2,
        ocu = False,
        grw = False
    ),
    "tempered_10d": dp_penalty.PenaltyParams(
        tau = 0.08,
        prop_sigma = np.array((0.008, 0.007, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001)) * 13,
        r_clip_bound = 2,
        ocu = False,
        grw = False
    ),
    "gauss_50d": dp_penalty.PenaltyParams(
        tau = 0.08,
        prop_sigma = np.hstack((np.array((0.008, 0.007)), np.repeat(0.001, 48))) * 0.8,
        r_clip_bound = 2,
        ocu = False,
        grw = False
    )
}
params = param_dict[args.experiment]

chain, accepts, clipped_r, iters = dp_penalty.dp_penalty(problem, epsilon, delta, params)
banana_util.save_results("dpps", args, problem, chain, accepts, clipped_r, None, iters, None)
