import numpy as np
import scipy.stats as stats
import arviz
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import abalone_lr as abalone

np.random.seed(43726482)
X_train, X_test, y_train, y_test = abalone.load_data()
n, dim = X_train.shape

iters = 10000
prop_sigma = np.zeros(9)
prop_sigma[:] = 0.065
prop_sigma[0] = 0.02
prop_sigma[1] = 0.02
prop_sigma[5] = 0.13
prop_sigma[6] = 0.13

clip_bound = np.inf
theta0 = np.zeros(dim)
p0 = 1

chain = np.zeros((iters + 1, dim))
chain[0] = theta0

with open("abalone_var_params.p", "rb") as file:
    var_params = pickle.load(file)
var_mu = var_params["mu"]
var_L = var_params["L"]
var_sigma = var_L @ var_L.transpose()
proposal_dist = stats.multivariate_normal(var_mu, var_sigma)

clipped = np.zeros(iters + 1)
accepts = 0
ltc = abalone.log_target(theta0, X_train, y_train)
for i in range(iters):
    current = chain[i, :]
    if np.random.rand() < p0 / (0.0 * i + 1):
        prop = proposal_dist.rvs(size=1)
        q = proposal_dist.logpdf(current) - proposal_dist.logpdf(prop)
    else:
        prop = stats.norm.rvs(size=dim, loc=current, scale=prop_sigma)
        q = 0

    ltp = abalone.log_target(prop, X_train, y_train)
    # ratio = llp - llc
    #
    # clip = clip_bound * np.sqrt(np.sum(current - prop)**2)
    # clipped[i + 1] = np.sum(np.abs(ratio) > clip)
    # ratio = np.clip(ratio, -clip, clip)
    #
    # lambd = np.sum(ratio) + lpp - lpc
    lambd = ltp - ltc + q
    u = np.log(np.random.rand())
    if u < lambd:
        chain[i + 1, :] = prop
        accepts += 1
        ltc = ltp
    else:
        chain[i + 1, :] = current
    if (i + 1) % 1000 == 0:
        print("Iteration: {}".format(i + 1))

print("Acceptance: {}".format(accepts / iters))
# print("Clipping: {}".format(np.sum(clipped) / iters / n))
fig, axes = plt.subplots(dim)
for i in range(dim):
    axes[i].plot(chain[:, i])
plt.show()

final_chain = chain[int((iters - 1) / 2) + 1:, :]

var_samples = proposal_dist.rvs(size=1000)
mcmc_trace = arviz.from_netcdf("abalone_post.nc")
mcmc_samples = np.concatenate(mcmc_trace.posterior.theta.values, axis=0)
fig, axes = plt.subplots(dim)
for i in range(dim):
    sns.kdeplot(mcmc_samples[:, i], ax=axes[i])
    sns.kdeplot(final_chain[:, i], ax=axes[i])
    sns.kdeplot(var_samples[:, i], ax=axes[i])
    # sns.kdeplot(alternative_samples[:, i], ax=axes[i])
    # axes[i].axvline(map[i], linestyle="dashed", color="black")
plt.show()

