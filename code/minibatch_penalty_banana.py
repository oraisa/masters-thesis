#!/usr/bin/env python3

import numpy as np
import banana_util
import dp_penalty_minibatch

args = banana_util.parse_args()
banana_util.set_seed(315731, args.index)

dim = args.dim
problem = banana_util.get_problem(args)
n, data_dim = problem.data.shape

epsilon = args.epsilon
delta = 0.1 / n

if args.tempering:
    if dim == 2:
        params = dp_penalty_minibatch.MinibatchPenaltyParams(
            tau = 0.9,
            prop_sigma = np.array((0.008, 0.007)) * 10,
            r_clip_bound = 2,
            batch_size = 1000,
            ocu = False,
            grw = False
        )
    elif dim == 10:
        params = dp_penalty_minibatch.MinibatchPenaltyParams(
            tau = 0.6,
            prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.0035,
            r_clip_bound = 1,
            batch_size = 1000,
            ocu = False,
            grw = False
        )
else:
    if dim == 2:
        params = dp_penalty_minibatch.MinibatchPenaltyParams(
            tau = 0.9,
            prop_sigma = np.array((0.008, 0.007)) * 0.2,
            r_clip_bound = 1,
            batch_size = 1000,
            ocu = False,
            grw = False
        )
    elif dim == 10:
        params = dp_penalty_minibatch.MinibatchPenaltyParams(
            tau = 0.6,
            prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.00003,
            r_clip_bound = 1,
            batch_size = 1000,
            ocu = False,
            grw = False
        )


chain, accepts, clipped_r, iters = dp_penalty_minibatch.dp_penalty_minibatch(problem, epsilon, delta, params)
banana_util.save_results("mdpps", args, problem, chain, accepts, clipped_r, None, iters, None)
