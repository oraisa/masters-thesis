import numpy as np 
import matplotlib.pyplot as plt
import mcmc_animation
import circle
import hmc

problem = circle.problem()
n, data_dim = problem.data.shape
dim = 2

epsilon = 2
delta = 1 / n

params = hmc.HMCParams(
    tau = 0.3,
    tau_g = 0.8,
    L = 50,
    eta = 0.09,
    mass = np.array((1, 1)),
    r_clip = 0.001,
    grad_clip = 0.0015,
)

res = hmc.hmc(
    problem, epsilon, delta, params
)

print("Acceptance: {}".format(res.acceptance))
print("Clipping: {}".format(res.clipped_r))
print("Grad Clipping: {}".format(res.clipped_grad))

print("Mean error: {}".format(np.sqrt(np.sum(np.mean(res.final_chain, axis=0)**2))))

# fig, axes = plt.subplots(dim)
# for i in range(dim):
#     axes[i].plot(chain[:, i])
# plt.show()

fig, ax = plt.subplots()
circle.plot_density(problem, ax)
ax.plot(res.final_chain[:, 0], res.final_chain[:, 1], color="orange", linestyle='', marker='.')
plt.show()
# plt.savefig("hmc_circle.png")

anim = mcmc_animation.MCMCAnimation(res.chain, res.leapfrog_chain)
circle.plot_density(problem, anim.ax)
# anim.save("hmc_circle.mp4")
anim.show()

