#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import mcmc_animation
import banana_model
import hmc
import util
import mmd

dim = 10
problem = banana_model.get_problem(dim, True)
# banana = banana_model.BananaModel(dim, a=20)
# data = banana.generate_test_data()
# posterior = banana.generate_posterior_samples(1000, data, 1)
# problem = util.Problem(banana.log_likelihood_per_sample, banana.log_prior, data, 1, posterior)
# problem = banana.get_problem()
n, data_dim = problem.data.shape
posterior = problem.true_posterior

epsilon = 4
delta = 0.1 / n

params = hmc.HMCParams(
    tau = 0.05,
    tau_g = 0.2,
    L = 10,
    eta = 0.01,
    mass = np.array((0.3, 1, 2, 2, 2, 2, 2, 2, 2, 2)),
    r_clip = 3,
    grad_clip = 3.0,
    theta0 = np.array((0.0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0))
)
# params.theta0 += np.random.normal(scale=0.00, size=dim)

chain, leapfrog_chain, accepts, clipped_r, iters, clipped_grad, grad_accesses = hmc.hmc(
    problem, epsilon, delta, params
)

final_chain = chain[int((iters - 1) / 2) + 1:, :]

print("Acceptance: {}".format(accepts / iters))
print("Clipping: {}".format(np.sum(clipped_r) / iters / n))
print("Grad Clipping: {}".format(clipped_grad / n / grad_accesses))
print("Grad accesses: {}".format(grad_accesses))

print("Mean error: {}".format(mmd.mean_error(final_chain, posterior)))
print("MMD: {}".format(mmd.mmd(np.asarray(final_chain), np.asarray(posterior))))


fig, axes = plt.subplots(dim)
for i in range(dim):
    axes[i].plot(chain[:, i])
plt.show()

fig, ax = plt.subplots()
# banana.plot_posterior(problem.data, ax, 1)
# banana.scatterplot_posterior(problem.data, ax, 1)
ax.scatter(posterior[:, 0], posterior[:, 1])
ax.plot(final_chain[:, 0], final_chain[:, 1], color="orange", linestyle='', marker='.')
plt.show()

anim = mcmc_animation.MCMCAnimation(chain, leapfrog_chain)
anim.ax.scatter(posterior[:, 0], posterior[:, 1], color="gray")
# banana.plot_posterior(problem.data, anim.ax, 1)
anim.show()
