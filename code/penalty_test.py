#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import banana_model
import gauss_model
import circle
import dp_penalty

# np.random.seed(53527482)

dim = 2
problem = banana_model.get_problem(dim=dim, a=20, n0=None, n=100000)
# problem = gauss_model.get_problem(dim=dim, n=200000)
# problem = circle.problem()
n, data_dim = problem.data.shape
true_posterior = problem.true_posterior

epsilon = 4
delta = 0.1 / n
params = dp_penalty.PenaltyParams(
    tau = 0.1,
    prop_sigma = np.repeat(0.008, 2),
    r_clip_bound = 1.8,
    ocu = True,
    grw = True

    # tau = 0.07,
    # prop_sigma = np.repeat(0.0002, dim),
    # r_clip_bound = 25,
    # ocu = False,
    # grw = False
    # tau = 0.1,
    # prop_sigma = np.repeat(0.0002, dim),
    # r_clip_bound = 45,
    # ocu = True,
    # grw = True
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

# fig, ax = plt.subplots()
# circle.plot_density(problem, ax)
# ax.plot(res.final_chain[:, 0], res.final_chain[:, 1], color="orange", linestyle='', marker='.')
# plt.show()

plt.scatter(true_posterior[:,0], true_posterior[:,1], alpha=0.2)
plt.scatter(res.final_chain[:, 0], res.final_chain[:, 1])
plt.show()
