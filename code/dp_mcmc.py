import banana_model
import dp_barker
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4673228)
dim = 10
problem = banana_model.get_problem(dim=dim, a=20, n0=None, n=200000)
n, data_dim = problem.data.shape

epsilon = 4
delta = 0.1 / n

params = dp_barker.BarkerParams(
    prop_sigma = np.repeat(8 * 0.0001, 10),
    batch_size = 2600
)

res = dp_barker.dp_barker(problem, epsilon, delta, params)


print("DP mcmc acceptance: {}".format(res.acceptance))
print(f"Clipped: {res.clipped_r}")
mean_error = res.mean_error
cov_error = res.cov_error
mmd = res.mmd
print(f"Mean error: {mean_error}")
print(f"Cov error: {cov_error}")
print(f"MMD: {mmd}")

fig, axes = plt.subplots(dim)
for i in range(dim):
    axes[i].plot(res.chain[:, i])
plt.show()

plt.scatter(problem.true_posterior[:,0], problem.true_posterior[:,1], alpha=0.2)
plt.scatter(res.final_chain[:, 0], res.final_chain[:, 1])
plt.show()
