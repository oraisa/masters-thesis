import banana_model
import mmd
import dp_barker
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4673228)
dim = 10
problem = banana_model.get_problem(dim, True)
n, data_dim = problem.data.shape

epsilon = 4
delta = 0.1 / n

params = dp_barker.BarkerParams(
    # prop_sigma = np.array((0.008, 0.007)) * 0.1,
    # batch_size = 1300
    prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.005,
    batch_size = 2600
)

chain, accepted, clip_count, iters = dp_barker.dp_barker(problem, epsilon, delta, params)

final_chain = chain[int((iters - 1) / 2) + 1:, :]
true_posterior = problem.true_posterior

print("DP mcmc acceptance: {}".format(accepted / iters))
print(f"Clipped: {np.sum(clip_count) / iters / params.batch_size}")
mean_error = mmd.mean_error(final_chain, true_posterior)
cov_error = mmd.cov_error(final_chain, true_posterior)
mmd = mmd.mmd(final_chain, true_posterior)
print(f"Mean error: {mean_error}")
print(f"Cov error: {cov_error}")
print(f"MMD: {mmd}")

fig, axes = plt.subplots(dim)
for i in range(dim):
    axes[i].plot(chain[:, i])
plt.show()

plt.scatter(true_posterior[:,0], true_posterior[:,1], alpha=0.2)
plt.scatter(final_chain[:, 0], final_chain[:, 1])
plt.show()
