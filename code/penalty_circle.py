import numpy as np
import matplotlib.pyplot as plt
import circle
import dp_penalty

problem = circle.problem()
n, data_dim = problem.data.shape
dim = 2

epsilon = 0.5
delta = 0.1 / n

params = dp_penalty.PenaltyParams(
    tau = 4,
    prop_sigma = np.repeat(0.3, dim),
    r_clip_bound = 0.002,
    ocu = True,
    grw = True
)

res = dp_penalty.dp_penalty(
    problem, epsilon, delta, params
)

print("Acceptance: {}".format(res.acceptance))
print("Clipping: {}".format(res.clipped_r))

# print("Mean error: {}".format(np.sqrt(np.sum(np.mean(res.final_chain, axis=0)**2))))
print("Mean error: {}".format(res.mean_error))

# fig, axes = plt.subplots(dim)
# for i in range(dim):
#     axes[i].plot(chain[:, i])
# plt.show()

fig, ax = plt.subplots()
circle.plot_density(problem, ax)
ax.plot(res.final_chain[:, 0], res.final_chain[:, 1], color="orange", linestyle='', marker='.')
plt.show()
# plt.savefig("hmc_circle.png")
