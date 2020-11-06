import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import banana_model as banana

np.random.seed(43726482)
data = banana.generate_test_data()

iters = 1000
prop_sigma = 0.01
clip_bound = 3
dim = 2
theta0 = np.array((0, 2.95))
chain = np.zeros((iters + 1, dim))
chain[0] = theta0

clipped = np.zeros(iters + 1)
accepts = 0
llc = banana.log_likelihood_no_sum(data, theta0)
for i in range(iters):
    current = chain[i, :]
    prop = stats.norm.rvs(size=dim, loc=current, scale=prop_sigma)

    lpc = banana.log_prior(current)
    lpp = banana.log_prior(prop)
    llp = banana.log_likelihood_no_sum(data, prop)
    ratio = llp - llc

    clip = clip_bound * np.sqrt(np.sum(current - prop)**2)
    clipped[i + 1] = np.sum(np.abs(ratio) > clip)
    ratio = np.clip(ratio, -clip, clip)

    lambd = np.sum(ratio) + lpp - lpc
    u = np.log(np.random.rand())
    if u < lambd:
        chain[i + 1, :] = prop
        accepts += 1
        llc = llp
    else:
        chain[i + 1, :] = current
    if (i + 1) % 100 == 0:
        print("Iteration: {}".format(i + 1))

print("Acceptance: {}".format(accepts / iters))
print("Clipping: {}".format(np.sum(clipped) / iters / 100000))
fig, axes = plt.subplots(2)
axes[0].plot(chain[:, 0])
axes[1].plot(chain[:, 1])
plt.show()

final_chain = chain[int((iters - 1) / 2) + 1:, :]
fig, axes = plt.subplots()
s1s, s2s, _ = banana.generate_posterior_samples(10000, data, 1)
axes.scatter(s1s, s2s, alpha=0.1)
axes.scatter(final_chain[:, 0], final_chain[:, 1])
plt.show()
