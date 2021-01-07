#!/usr/bin/env python3

import numpy as np
import banana_util
import dp_barker

args = banana_util.parse_args()
banana_util.set_seed(10478, args.index)

dim = args.dim
problem = banana_util.get_problem(args)
n, data_dim = problem.data.shape

epsilon = args.epsilon
delta = 0.1 / n

if args.tempering:
    if dim == 2:
        params = dp_barker.BarkerParams(
            prop_sigma = np.array((0.008, 0.007)) * 15,
            batch_size = 1300
        )
    elif dim == 10:
        params = dp_barker.BarkerParams(
            prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.005,
            batch_size = 2600
        )
else:
    if dim == 2:
        params = dp_barker.BarkerParams(
            prop_sigma = np.array((0.008, 0.007)) * 0.1,
            batch_size = 1300
        )
    elif dim == 10:
        params = dp_barker.BarkerParams(
            prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.0001,
            batch_size = 2600
        )


chain, accepts, clipped_r, iters = dp_barker.dp_barker(problem, epsilon, delta, params)
banana_util.save_results("barker", args, problem, chain, accepts, clipped_r, None, iters, None)
