#!/usr/bin/env python3
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import banana_model
import circle
import mmd
import dp_penalty
import util

# np.random.seed(53527482)

dim = 50
problem = banana_model.get_problem(dim=dim, a=0, n0=None, n=200000)
n, data_dim = problem.data.shape
true_posterior = problem.true_posterior

epsilon = 4
delta = 0.1 / n
params = dp_penalty.PenaltyParams(
    # tau = 0.1,
    # prop_sigma = np.array((0.008, 0.007)) * 1,
    # r_clip_bound = 3,
    # ocu = False,
    # grw = False

    tau = 0.06,
    prop_sigma = np.hstack((np.array((8, 7)), np.repeat(5, 48))) * 0.00013,
    r_clip_bound = 3,
    ocu = False,
    grw = False
)


res = dp_penalty.dp_penalty(problem, epsilon, delta, params)

print("Acceptance: {}".format(res.acceptance))
print("Clipping: {}".format(res.clipped_r))

print("Mean error: {}".format(res.mean_error))
print("Cov error: {}".format(res.cov_error))
print("MMD: {}".format(res.mmd))

fig, axes = plt.subplots(min(dim, 10))
for i in range(min(dim, 10)):
    axes[i].plot(res.chain[:, i])
plt.show()

plt.scatter(true_posterior[:,0], true_posterior[:,1], alpha=0.2)
plt.scatter(res.final_chain[:, 0], res.final_chain[:, 1])
plt.show()
