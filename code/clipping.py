import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import banana_model as banana
import mmd

np.random.seed(43726482)
data = banana.generate_test_data()
n, data_dim = data.shape

iters = 1000
prop_sigma = 0.01
clip_bound = 10
dim = 2
theta0 = np.array((0, 2.95))
chain = np.zeros((iters + 1, dim))
chain[0] = theta0

clipped = np.zeros(iters + 1)
clip_diff = np.zeros(iters + 1)
orig_ratios = np.zeros(iters + 1)
accepts = 0
diff_accepts = np.zeros(iters + 1)
llc = banana.log_likelihood_no_sum(data, theta0)
for i in range(iters):
    current = chain[i, :]
    prop = stats.norm.rvs(size=dim, loc=current, scale=prop_sigma)

    lpc = banana.log_prior(current)
    lpp = banana.log_prior(prop)
    llp = banana.log_likelihood_no_sum(data, prop)
    ratio = llp - llc
    orig_ratio = ratio
    orig_ratios[i + 1] = np.sum(orig_ratio)

    clip = clip_bound * np.sqrt(np.sum(current - prop)**2)
    clipped[i + 1] = np.sum(np.abs(ratio) > clip)
    ratio = np.clip(ratio, -clip, clip)
    clip_diff[i + 1] = np.sum(orig_ratio - ratio)

    lambd = np.sum(ratio) + lpp - lpc
    u = np.log(np.random.rand())
    accept = u < lambd
    orig_accept = u < np.sum(orig_ratio) + lpp - lpc
    if accept != orig_accept:
        diff_accepts[i + 1] = 1
    if accept:
        chain[i + 1, :] = prop
        accepts += 1
        llc = llp
    else:
        chain[i + 1, :] = current
    if (i + 1) % 100 == 0:
        print("Iteration: {}".format(i + 1))

final_chain = chain[int((iters - 1) / 2) + 1:, :]
print("Acceptance: {}".format(accepts / iters))
print("Clipping: {}".format(np.sum(clipped) / iters / n))
print("Different decisions: {}".format(np.sum(diff_accepts)))
s1s, s2s, _ = banana.generate_posterior_samples(1000, data, 1)
posterior = np.vstack((s1s, s2s)).transpose()
print("MMD: {}".format(mmd.mmd(final_chain, posterior)))
s1s, s2s, _ = banana.generate_posterior_samples(1000, data, 1)
alt_posterior = np.vstack((s1s, s2s)).transpose()
print("Base MMD: {}".format(mmd.mmd(alt_posterior, posterior)))

fig, axes = plt.subplots(2)
axes[0].plot(chain[:, 0])
axes[1].plot(chain[:, 1])
plt.show()

fig, axes = plt.subplots()
s1s, s2s, _ = banana.generate_posterior_samples(10000, data, 1)
axes.scatter(s1s, s2s, alpha=0.1)
axes.scatter(final_chain[:, 0], final_chain[:, 1])
plt.show()

fig, axes = plt.subplots(1, 2)
inds = np.arange(iters + 1)[diff_accepts == 1]
for i in inds:
    axes[0].axvline(i, color="red")
axes[0].plot(clip_diff)

for i in inds:
    axes[1].axvline(i, color="red")
axes[1].plot(orig_ratios)
plt.show()

