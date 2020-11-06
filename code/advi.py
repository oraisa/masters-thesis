import numpy as npa
import jax.numpy as np 
import jax
import scipy.stats as stats
import matplotlib.pyplot as plt
import banana_model as banana

npa.random.seed(4368627)
data = banana.generate_test_data()
n, dim = data.shape

@jax.jit
def log_target(theta, subsample):
    return n / subsample.size * banana.log_likelihood(data[subsample], theta) + banana.log_prior(theta)

log_target_grad = jax.grad(log_target)

@jax.jit
def advi():
    iters = 20000
    mu_chain = np.zeros((iters, dim))
    L_chain = np.zeros((iters, dim))
    mu = np.zeros(dim)
    L = np.ones(dim)
    rho0 = 0.5
    b = 1000
    mu_ada_grad = np.zeros(dim)
    L_ada_grad = np.zeros(dim)

    for i in range(iters):
        eta = stats.norm.rvs(size=2)
        theta = L * eta + mu

        subsample = npa.random.choice(n, b)
        mu_grad = log_target_grad(theta, subsample)
        L_grad = log_target_grad(theta, subsample) * eta + 1 / L
        mu_ada_grad += mu_grad**2
        L_ada_grad += L_grad**2
        rho_mu = rho0 / (np.sqrt(mu_ada_grad) + 0.0000001)
        rho_L = rho0 / (np.sqrt(L_ada_grad) + 0.0000001)

        mu += rho_mu * mu_grad
        L += rho_L * L_grad
        mu_chain = jax.ops.index_update(mu_chain, i, mu)
        L_chain = jax.ops.index_update(L_chain, i, L)

        if (i + 1) % 1000 == 0:
            print("Iteration: {}".format(i + 1))
    return mu, L, mu_chain, L_chain

mu, L, mu_chain, L_chain = advi()

fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(mu_chain[:, 0])
axes[0, 1].plot(mu_chain[:, 1])
axes[1, 0].plot(L_chain[:, 0])
axes[1, 1].plot(L_chain[:, 1])
plt.show()

posterior_sample = stats.norm.rvs(size=(1000, 2), loc=mu, scale=np.abs(L))
s1s, s2s, _ = banana.generate_posterior_samples(5000, data, 1)
plt.scatter(s1s, s2s, alpha=0.1)
plt.scatter(posterior_sample[:, 0], posterior_sample[:, 1])
plt.show()
