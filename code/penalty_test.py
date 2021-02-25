#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import experiments
import dp_penalty

# np.random.seed(53527482)

problem = experiments.experiments["hard-2d"].get_problem()
dim = problem.dim
n, data_dim = problem.data.shape
true_posterior = problem.true_posterior

epsilon = 4
delta = 0.1 / n
params = dp_penalty.PenaltyParams(
    tau = 0.1,
    prop_sigma = np.repeat(0.0015, 2),
    r_clip_bound = 8.5,
    ocu = True,
    grw = True
)

res = dp_penalty.dp_penalty(problem, problem.get_start_point(0), epsilon, delta, params)

print("Acceptance: {}".format(res.acceptance))
print("Clipping: {}".format(res.clipped_r))

print("Mean error: {}".format(res.mean_error))
print("Cov error: {}".format(res.cov_error))
print("MMD: {}".format(res.mmd))

fig, axes = plt.subplots(min(dim, 10))
for i in range(min(dim, 10)):
    axes[i].plot(res.chain[:, i])
plt.show()

# fig, ax = plt.subplots()
# circle.plot_density(problem, ax)
# ax.plot(res.final_chain[:, 0], res.final_chain[:, 1], color="orange", linestyle='', marker='.')
# plt.show()

plt.scatter(true_posterior[:,0], true_posterior[:,1], alpha=0.2)
plt.scatter(res.chain[:, 0], res.chain[:, 1])
start = problem.get_start_point(0)
plt.plot(start[0], start[1], color="red", linestyle="", marker=".")
plt.show()
