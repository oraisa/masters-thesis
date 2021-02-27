#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import banana_model
import experiments
import mmd
import dp_penalty_minibatch
import util

# np.random.seed(53726482)

problem = experiments.experiments["easy-10d"].get_problem()
dim = problem.dim
# problem = banana_model.get_problem(dim=dim, a=20, n0=1000, n=200000)
n, data_dim = problem.data.shape

epsilon = 1
delta = 0.1 / n
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 0.5,
    prop_sigma = np.repeat(8 * 0.00004, 10),
    r_clip_bound = 2.0,
    batch_size = 1000,
    ocu = True,
    grw = True
)

res = dp_penalty_minibatch.dp_penalty_minibatch(
    problem, problem.get_start_point(0), epsilon, delta, params
)

print("Acceptance: {}".format(res.acceptance))
print("Clipping: {}".format(res.clipped_r))

print("Mean error: {}".format(res.mean_error))
print("Cov error: {}".format(res.cov_error))
print("MMD: {}".format(res.mmd))

fig, axes = plt.subplots(dim)
for i in range(dim):
    axes[i].plot(res.chain[:, i])
plt.show()

plt.scatter(problem.true_posterior[:,0], problem.true_posterior[:,1], alpha=0.2)
plt.scatter(res.final_chain[:, 0], res.final_chain[:, 1])
plt.show()
