import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import circle
import hmc

problem = circle.problem()
n, data_dim = problem.data.shape
dim = 2

epsilon = 3
delta = 1 / n

params = hmc.HMCParams(
    tau = 0.1,
    tau_g = 0.8,
    L = 50,
    eta = 0.1,
    mass = np.array((1, 1)),
    r_clip = 0.001,
    grad_clip = 0.001,
    theta0 = np.array((0.0, 3.0))
)

chain, leapfrog_chain, accepts, clipped_r, iters, clipped_grad, grad_accesses = hmc.hmc(
    problem, epsilon, delta, params
)

print("Acceptance: {}".format(accepts / iters))
print("Clipping: {}".format(np.sum(clipped_r) / iters / n))
print("Grad Clipping: {}".format(clipped_grad / n / grad_accesses))

final_chain = chain[int((iters - 1) / 2) + 1:, :]
print("Mean error: {}".format(np.sqrt(np.sum(np.mean(final_chain, axis=0)**2))))

# fig, axes = plt.subplots(dim)
# for i in range(dim):
#     axes[i].plot(chain[:, i])
# plt.show()

# plt.scatter(s1s, s2s, alpha=0.1)
fig, ax = plt.subplots()
circle.plot_density(problem, ax)
ax.plot(final_chain[:, 0], final_chain[:, 1], color="orange", linestyle='', marker='.')
plt.show()

fig, ax = plt.subplots()
circle.plot_density(problem, ax)
line = ax.plot([], [])[0]
sample_line = ax.plot((), (), linestyle='', marker='.')[0]
def animate(i):
    min_i = np.max((0, i - i % params.L))
    line.set_data(leapfrog_chain[min_i:i, 0], leapfrog_chain[min_i:i, 1])
    sample_index = int(i / params.L)
    sample_line.set_data(chain[:sample_index, 0], chain[:sample_index, 1])
    return [line, sample_line]

anim = animation.FuncAnimation(
    fig, animate, leapfrog_chain.shape[0], interval=20, blit=True
)
plt.show()

