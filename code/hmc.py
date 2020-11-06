import numpy as np
import scipy.stats as stats 
import banana_model as banana
import arviz 
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(43726482)
# X_train, X_test, y_train, y_test = abalone.load_data()
# n, dim = X_train.shape
data = banana.generate_test_data()
n, dim = data.shape


epsilon = 5
delta = 1 / n

tau = 0.1
tau_g = 0.8
L = 8
eta = 0.0008
r_clip_bound = 2
grad_clip = 0.3

rho = (np.sqrt(epsilon - np.log(delta)) - np.sqrt(-np.log(delta)))**2
rho_l = 1 / (2 * tau**2 * n)
rho_g = (L + 1) / (2 * tau_g**2 * n)
print("rho_l: {}".format(rho_l))
print("rho_g: {}".format(rho_g))

iters = int(rho / (rho_l + rho_g))
print("Iterations: {}".format(iters))

sigma = tau * np.sqrt(n)
# iters = 1000
theta0 = np.zeros(dim)
theta0 = np.array((0, 2.98))

chain = np.zeros((iters + 1, dim))
chain[0, :] = theta0
clipped_r = np.zeros(iters)
clipped_grad = np.zeros(iters)
accepts = 0

def grad_fun(theta, i):
    ll_grads = banana.log_likelihood_grads(data, theta)
    norms = np.linalg.norm(ll_grads, axis=1, ord=2, keepdims=True)
    not_to_clip = norms < grad_clip
    norms[not_to_clip] = grad_clip
    ll_grads = ll_grads / norms * grad_clip
    clipped_grad[i] =  n - not_to_clip.sum()

    pri_grad = banana.log_prior_grad(theta)
    noise_sigma = 2 * tau_g * np.sqrt(n) * grad_clip
    return np.sum(ll_grads, axis=0) + pri_grad + stats.norm.rvs(size=dim, scale=noise_sigma)

grad = grad_fun(theta0, 0)
llc = banana.log_likelihood_no_sum(data, theta0)
for i in range(iters):
    current = chain[i, :]
    p = stats.norm.rvs(size=dim)
    p_orig = p
    prop = current
    grad_new = grad

    for j in range(L):
        p += eta * grad_new * 0.5
        prop += eta * p
        grad_new = grad_fun(prop, 0)
        p += eta * grad_new * 0.5

    llp = banana.log_likelihood_no_sum(data, prop)
    r = llp - llc
    d = np.sqrt(np.sum((current - prop)**2))
    clip = d * r_clip_bound
    clipped_r[i] = np.sum(np.abs(r) > clip)
    r = np.clip(r, -clip, clip)

    lpp = banana.log_prior(prop)
    lpc = banana.log_prior(current)

    s = stats.norm.rvs(size=1, scale=sigma * d * 2 * r_clip_bound)
    dH = 0.5 * np.sum(p_orig**2) - 0.5 * np.sum(p**2) + np.sum(r) + lpp - lpc + s
    u = np.log(np.random.rand())

    if u < dH - 0.5 * (sigma * d * 2 * r_clip_bound)**2:
        chain[i + 1, :] = prop 
        grad = grad_new 
        llc = llp 
        accepts += 1
    else:
        chain[i + 1, :] = current
    if (i + 1) % 100 == 0:
        print("Iteration: {}".format(i + 1))


print("Acceptance: {}".format(accepts / iters))
print("Clipping: {}".format(np.sum(clipped_r) / iters / n))
print("Grad Clipping: {}".format(np.sum(clipped_grad) / iters / n))
fig, axes = plt.subplots(dim)
for i in range(dim):
    axes[i].plot(chain[:, i])
plt.show()

final_chain = chain[int((iters - 1) / 2) + 1:, :]

s1s, s2s, _ = banana.generate_posterior_samples(10000, data, 1)
plt.scatter(s1s, s2s, alpha=0.1)
plt.scatter(final_chain[:, 0], final_chain[:, 1])
plt.show()

# mcmc_trace = arviz.from_netcdf("abalone_post.nc")
# mcmc_samples = np.concatenate(mcmc_trace.posterior.theta.values, axis=0)
# fig, axes = plt.subplots(dim)
# for i in range(dim):
#     sns.kdeplot(mcmc_samples[:, i], ax=axes[i])
#     sns.kdeplot(final_chain[:, i], ax=axes[i])
# plt.show()

