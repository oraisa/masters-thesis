import jax
import jax.numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import arviz
import jax.scipy.stats as stats 
import abalone_lr as abalone

X_train, X_test, y_train, y_test = abalone.load_data()
n, dim = X_train.shape

def advi(iters, rho0, log_target_grad, dim):
    mu_chain = np.zeros((iters, dim))
    L_chain = np.zeros((iters, dim))
    mu = np.zeros(dim)
    L = np.ones(dim)
    mu_ada_grad = np.zeros(dim)
    L_ada_grad = np.zeros(dim)
    key = jax.random.PRNGKey(423786)

    for i in range(iters):
        key, subkey = jax.random.split(key)
        eta = jax.random.normal(subkey, shape=(dim,))
        theta = L * eta + mu

        mu_grad = log_target_grad(theta)
        L_grad = log_target_grad(theta) * eta + 1 / L
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

@jax.jit
def log_target_grad(theta):
    return abalone.log_target_grad(theta, X_train, y_train)


mu, L, mu_chain, L_chain = advi(
    10000, 0.1, log_target_grad, dim
)

fig, axes = plt.subplots(2, dim)
for i in range(dim):
    axes[0, i].plot(mu_chain[:, i])
for i in range(dim):
    axes[1, i].plot(L_chain[:, i])
plt.show()

mcmc_trace = arviz.from_netcdf("abalone_post.nc")
mcmc_samples = np.concatenate(mcmc_trace.posterior.theta.values, axis=0)

xs = np.linspace(-2, 2)
fig, axes = plt.subplots(dim)
for i in range(dim):
    sns.kdeplot(mcmc_samples[:, i], ax=axes[i])
    axes[i].plot(xs+mu[i], stats.norm.pdf(xs+mu[i], loc=mu[i], scale=np.abs(L[i])))
plt.tight_layout()
plt.show()
