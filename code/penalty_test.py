#!/usr/bin/env python3
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import banana_model
import mmd
import dp_penalty
import util

np.random.seed(53726482)

dim = 2
banana = banana_model.BananaModel(dim=dim, a=20)
problem = banana.get_problem()
n, data_dim = problem.data.shape

epsilon = 4
delta = 0.1 / n
params = dp_penalty.PenaltyParams(
    tau = 0.1,
    prop_sigma = np.array((0.008, 0.007)),
    theta0 = np.array((0.0, 3.0)),
    r_clip_bound = 2
)
params.theta0 += np.random.normal(scale=0.02, size=dim)


chain, accepts, clipped_r, iters = dp_penalty.dp_penalty(problem, epsilon, delta, params)

print("Acceptance: {}".format(accepts / iters))
print("Clipping: {}".format(np.sum(clipped_r) / iters / n))

final_chain = chain[int((iters - 1) / 2) + 1:, :]
true_posterior = banana.generate_posterior_samples(1000, problem.data, 1)
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
