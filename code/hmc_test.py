import numpy as np
import matplotlib.pyplot as plt
import mcmc_animation
import banana_model
import hmc
import util
import mmd

dim = 2
problem = banana_model.get_problem(dim=dim, a=40, n0=None, n=100000)
n, data_dim = problem.data.shape
posterior = problem.true_posterior

epsilon = 4
delta = 0.1 / n

params = hmc.HMCParams(
    tau = 0.08,
    tau_g = 0.35,
    L = 20,
    eta = 0.00030,
    mass = np.array((0.5, 1)),
    r_clip = 2.1,
    grad_clip = 2.5,

    # tau = 0.05,
    # tau_g = 0.10,
    # L = 5,
    # eta = 0.0002,
    # mass = np.hstack((np.array((0.1, 1)), np.repeat(2, dim - 2))),
    # r_clip = 3.1,
    # grad_clip = 6.0,
)
# problem.theta0 = np.array((-0.03, 2.95))

result = hmc.hmc(
    problem, epsilon, delta, params
)
final_chain = result.final_chain

print("Acceptance: {}".format(result.acceptance))
print("Clipping: {}".format(result.clipped_r))
print("Grad Clipping: {}".format(result.clipped_grad))
# print("Grad accesses: {}".format(grad_accesses))

print("Mean error: {}".format(result.mean_error))
print("Cov error: {}".format(result.cov_error))
print("MMD: {}".format(result.mmd))


fig, axes = plt.subplots(min(dim, 10))
for i in range(min(dim, 10)):
    axes[i].plot(result.chain[:, i])
plt.show()

fig, ax = plt.subplots()
# banana.plot_posterior(problem.data, ax, 1)
# banana.scatterplot_posterior(problem.data, ax, 1)
ax.scatter(posterior[:, 0], posterior[:, 1])
ax.plot(final_chain[:, 0], final_chain[:, 1], color="orange", linestyle='', marker='.')
plt.show()

anim = mcmc_animation.MCMCAnimation(result.chain, result.leapfrog_chain)
anim.ax.scatter(posterior[:, 0], posterior[:, 1], color="gray")
# banana.plot_posterior(problem.data, anim.ax, 1)
anim.show()
