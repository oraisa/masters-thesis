import numpy as np
import scipy.stats as stats 
import banana_model as banana
import arviz 
import seaborn as sns
import matplotlib.pyplot as plt
import mmd

np.random.seed(53726482)
# X_train, X_test, y_train, y_test = abalone.load_data()
# n, dim = X_train.shape
data = banana.generate_test_data()
n, dim = data.shape

epsilon = 5
delta = 1 / n
tau = 0.1
rho = (np.sqrt(epsilon - np.log(delta)) - np.sqrt(-np.log(delta)))**2
iters = int(2 * tau**2 * n * rho)
print("Iterations: {}".format(iters))

sigma = tau * np.sqrt(n)
# iters = 1000
prop_sigma = np.array((0.008, 0.007))#, 0.001))
theta0 = np.zeros(dim)
theta0[1] = 2.98
r_clip_bound = 2

chain = np.zeros((iters + 1, dim))
chain[0, :] = theta0
clipped_r = np.zeros(iters)
accepts = 0

llc = banana.log_likelihood_no_sum(data, theta0)
for i in range(iters):
    current = chain[i, :]
    prop = stats.norm.rvs(size=dim, loc=current, scale=prop_sigma)

    llp = banana.log_likelihood_no_sum(data, prop)
    r = llp - llc
    d = np.sqrt(np.sum((current - prop)**2))
    clip = d * r_clip_bound
    clipped_r[i] = np.sum(np.abs(r) > clip)
    r = np.clip(r, -clip, clip)

    lpp = banana.log_prior(prop)
    lpc = banana.log_prior(current)

    s = stats.norm.rvs(size=1, scale=sigma * d * 2 * r_clip_bound)
    lambd = np.sum(r) + lpp - lpc + s
    u = np.log(np.random.rand())

    if u < lambd - 0.5 * (sigma * d * 2 * r_clip_bound)**2:
        chain[i + 1, :] = prop 
        llc = llp 
        accepts += 1
    else:
        chain[i + 1, :] = current
    if (i + 1) % 100 == 0:
        print("Iteration: {}".format(i + 1))


print("Acceptance: {}".format(accepts / iters))
print("Clipping: {}".format(np.sum(clipped_r) / iters / n))

final_chain = chain[int((iters - 1) / 2) + 1:, :]
s1s, s2s, srest = banana.generate_posterior_samples(5000, data, 1)
true_posterior = np.hstack((s1s.reshape(-1, 1), s2s.reshape(-1, 1), srest))
print("Mean error: {}".format(mmd.mean_error(final_chain, true_posterior)))
print("Cov error: {}".format(mmd.cov_error(final_chain, true_posterior)))
print("MMD: {}".format(mmd.mmd(final_chain, true_posterior)))

fig, axes = plt.subplots(dim)
for i in range(dim):
    axes[i].plot(chain[:, i])
plt.show()

plt.scatter(s1s, s2s, alpha=0.1)
plt.scatter(final_chain[:, 0], final_chain[:, 1])
plt.show()


