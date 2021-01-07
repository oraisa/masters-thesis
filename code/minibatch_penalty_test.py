#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import banana_model
import mmd
import dp_penalty_minibatch
import util

np.random.seed(53726482)

dim = 2
problem = banana_model.get_problem(dim, True)
n, data_dim = problem.data.shape

epsilon = 4
delta = 0.1 / n
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 0.9,
    prop_sigma = np.array((0.008, 0.007)) * 10,
    r_clip_bound = 1,
    batch_size = 1000,
    ocu = True,
    grw = True

    # tau = 0.6,
    # prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.00006,
    # r_clip_bound = 1,
    # batch_size = 1000,
    # ocu = True,
    # grw = True
)

chain, accepts, clipped_r, iters = dp_penalty_minibatch.dp_penalty_minibatch(problem, epsilon, delta, params)

print("Acceptance: {}".format(accepts / iters))
print("Clipping: {}".format(np.sum(clipped_r) / iters / n))

final_chain = chain[int((iters - 1) / 2) + 1:, :]
true_posterior = problem.true_posterior
print("Mean error: {}".format(mmd.mean_error(final_chain, true_posterior)))
print("Cov error: {}".format(mmd.cov_error(final_chain, true_posterior)))
print("MMD: {}".format(mmd.mmd(final_chain, true_posterior)))

fig, axes = plt.subplots(dim)
for i in range(dim):
    axes[i].plot(chain[:, i])
plt.show()

plt.scatter(true_posterior[:,0], true_posterior[:,1], alpha=0.2)
plt.scatter(final_chain[:, 0], final_chain[:, 1])
plt.show()
