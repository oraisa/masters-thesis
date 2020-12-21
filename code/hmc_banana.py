#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mcmc_animation
import banana_model
import hmc
import util
import mmd

dim = 2
banana = banana_model.BananaModel(dim, a=20)
data = banana.generate_test_data()
problem = util.Problem(banana.log_likelihood_per_sample, banana.log_prior, data)
n, data_dim = problem.data.shape

epsilon = 3
delta = 1 / n

params = hmc.HMCParams(
    tau = 0.1,
    tau_g = 0.4,
    L = 10,
    eta = 0.0005,
    mass = np.array((0.5, 1)),
    r_clip = 2,
    grad_clip = 1.0,
    theta0 = np.array((0.0, 3.0))
)

chain, leapfrog_chain, accepts, clipped_r, iters, clipped_grad, grad_accesses = hmc.hmc(
    problem, epsilon, delta, params
)

print("Acceptance: {}".format(accepts / iters))
print("Clipping: {}".format(np.sum(clipped_r) / iters / n))
print("Grad Clipping: {}".format(clipped_grad / n / grad_accesses))
print("Grad accesses: {}".format(grad_accesses))

final_chain = chain[int((iters - 1) / 2) + 1:, :]
posterior = banana.generate_posterior_samples(1000, data, 1)
print("Mean error: {}".format(mmd.mean_error(final_chain, posterior)))
print("MMD: {}".format(mmd.mmd(np.asarray(final_chain), np.asarray(posterior))))

fig, axes = plt.subplots(dim)
for i in range(dim):
    axes[i].plot(chain[:, i])
plt.show()

fig, ax = plt.subplots()
banana.plot_posterior(data, ax, 1)
banana.scatterplot_posterior(data, ax, 1)
ax.plot(final_chain[:, 0], final_chain[:, 1], color="orange", linestyle='', marker='.')
plt.show()

anim = mcmc_animation.MCMCAnimation(chain, leapfrog_chain)
banana.plot_posterior(data, anim.ax, 1)
anim.show()
