import jax.numpy as np
import jax
import jax.scipy.stats as stats
import scipy.stats as scipystats
import matplotlib.pyplot as plt
import numpy.random as npr

tau0 = 0.001
tau1 = 0.05
tau2 = 0.4
tau3 = 10
sigma1 = 1 / np.sqrt(tau1)
sigma2 = 1 / np.sqrt(tau2)
sigma3 = 1 / np.sqrt(tau3)
# a = 80.0
# a = 40
a = 20
b = 0.0
m = 0.0
dim = 2

@jax.jit
def log_prior(theta):
    return (
        stats.norm.logpdf(theta[0], scale=1 / np.sqrt(tau0))
        + stats.norm.logpdf(
            theta[1] + a * (theta[0] - m)**2 + b, scale=1 / np.sqrt(tau0)
        )
        + np.sum(stats.norm.logpdf(theta[2:], scale=1 / np.sqrt(tau0)))
    )

logc1 = -np.log(sigma1 * np.sqrt(2 * np.pi))
logc2 = -np.log(sigma2 * np.sqrt(2 * np.pi))
logc3 = -np.log(sigma3 * np.sqrt(2 * np.pi))

@jax.jit 
def log_likelihood_per_sample(x, theta):
    theta1 = theta[0]
    theta2 = theta[1]
    theta_rest = theta[2:]
    x1 = x[0]
    x2 = x[1]
    xrest = x[2:]
    term1 = -0.5 * (x1 - theta1)**2 / sigma1**2 + logc1
    term2 = -0.5 * (x2 - (theta2 + a * (theta1 - m)**2) + b)**2 / sigma2**2 + logc2
    term3 = -0.5 * np.sum((xrest - theta_rest)**2) / sigma3**2 + logc3 * (dim - 2)
    return term1 + term2 + term3

log_likelihood_no_sum = jax.jit(jax.vmap(log_likelihood_per_sample, in_axes=(0, None)))
log_likelihood = jax.jit(lambda X, theta: np.sum(log_likelihood_no_sum(X, theta)))
# @jax.jit
# def log_likelihood_no_sum(X, theta):
#     theta1 = theta[0]
#     theta2 = theta[1]
#     return (
#         -0.5 * (X[...,0] - theta1)**2 / sigma1**2 + logc1 
#         - 0.5 * (X[...,1] - (theta2 + a * (theta1 - m)**2) + b)**2 / sigma2**2
#         + logc2
#         - 0.5 * np.sum((X[...,2:] - theta[2:])**2, axis=1) / sigma3**2 + logc3 * (dim - 2)
#     )
#
# @jax.jit
# def log_likelihood(X, theta):
#     theta1 = theta[0]
#     theta2 = theta[1]
#     return (
#         np.sum(-0.5 * (X[...,0] - theta1)**2 / sigma1**2 + logc1)
#         - np.sum(0.5 * (X[...,1] - (theta2 + a * (theta1 - m)**2) + b)**2 / sigma2**2 + logc2)
#         - np.sum(0.5 * (X[...,2:] - theta[2:])**2 / sigma3**2 + logc3)
#     )
#     # return np.sum(log_likelihood_no_sum(X, theta))

log_likelihood_grads = jax.jit(jax.vmap(jax.grad(log_likelihood_per_sample, 1), in_axes=(0, None)))
log_prior_grad = jax.jit(jax.grad(log_prior))

@jax.jit
def log_likelihood_grad_clipped(data, theta, clip):
    n, dim = data.shape
    grads = log_likelihood_grads(data, theta)
    grads, did_clip = jax.vmap(clip_norm, in_axes=(0, None))(grads, clip)
    clipped_grad = np.sum(did_clip)
    return (np.sum(grads, axis=0), clipped_grad)

@jax.jit 
def clip_norm(x, bound):
    norm = np.sqrt(np.sum(x**2))
    clipped_norm = np.max(np.array((norm, bound)))
    return (x / norm * clipped_norm, norm > bound)

def generate_test_data():
    n = 100000
    theta1 = 0
    theta2 = 3
    theta_rest = np.zeros(dim - 2)
    # theta_rest = jax.ops.index_update(theta_rest, 0, 5)
    # theta_rest = jax.ops.index_update(theta_rest, 2, 10)

    x1s = scipystats.norm.rvs(size=n, loc=theta1, scale=sigma1)
    x2s = scipystats.norm.rvs(size=n, loc=theta2 + a * (theta1 - m)**2 + b, scale=sigma2)
    xrest = scipystats.norm.rvs(
        size=n*(dim - 2), loc=np.repeat(theta_rest, n), scale=sigma3
    ).reshape((dim - 2, n))
    return np.transpose(np.vstack((x1s, x2s, xrest)))

def banana_density(theta1, theta2, mu1, mu2, sigma1, sigma2, a, b, m):
    return (
        stats.norm.pdf(theta1, loc=mu1, scale=sigma1)
        * stats.norm.pdf(theta2 + a * (theta1 - m)**2 + b, loc=mu2, scale=sigma2)
    )

def banana_g2(x1, x2, m):
    return x2 - a * (x1 - m)**2 - b

def compute_posterior_params(X, T):
    n = X.shape[0]
    mu1 = (T * n * tau1 * X[..., 0].mean()) / (T * n * tau1 + tau0)
    mu2 = (T * n * tau2 * X[..., 1].mean()) / (T * n * tau2 + tau0)
    murest = (T * n * tau3 * X[...,2:].mean(axis=0)) / (T * n * tau3 + tau0)
    sigma1_p = 1 / np.sqrt(T * n * tau1 + tau0)
    sigma2_p = 1 / np.sqrt(T * n * tau2 + tau0)
    sigma3_p = 1 / np.sqrt(T * n * tau3 + tau0)
    return (mu1, mu2, murest, sigma1_p, sigma2_p, sigma3_p)

def generate_posterior_samples(n, X, T):
    mu1, mu2, murest, sigma1_p, sigma2_p, sigma3_p = compute_posterior_params(X, T)
    s1s = scipystats.norm.rvs(size=n, loc=mu1, scale=sigma1_p)
    s2s = banana_g2(s1s, scipystats.norm.rvs(size=n, loc=mu2, scale=sigma2_p), m)
    srest = scipystats.norm.rvs(
        size=n * (dim - 2), loc=np.repeat(murest, n), scale=sigma3_p
    ).reshape((dim - 2), n).transpose()
    return (s1s, s2s, srest)

def scatterplot_posterior(X, ax, T):
    s1s, s2s, _ = generate_posterior_samples(1000, X, T)
    ax.scatter(s1s, s2s)

def plot_posterior(X, ax, T):
    mu1, mu2, _, sigma1_p, sigma2_p, __ = compute_posterior_params(X, T)
    xs = np.linspace(-0.1, 0.1, 1000)
    ys = np.linspace(-0.1, 0.1, 1000) + mu2
    X, Y = np.meshgrid(xs, ys)
    Z = banana_density(X, Y, mu1, mu2, sigma1_p, sigma2_p, a, b, m)
    ax.contour(X, Y, Z)

if __name__ == "__main__":
    T = 100 / 100000
    npr.seed(463728)

    X = generate_test_data()

    fig, ax = plt.subplots()
    scatterplot_posterior(X, ax, 1)
    plot_posterior(X, ax, 1)
    plt.show()
    # plt.savefig("../latex/figures/banana-nontempered/posterior-easy.pdf")

    # fig, ax = plt.subplots()
    # scatterplot_posterior(X, ax, T)
    # plot_posterior(X, ax, T)
    # plt.show()
    # plt.savefig("../latex/figures/banana-tempered/posterior-easy.pdf")

    s1s, s2s, srest = generate_posterior_samples(10000, X, 1)
    print(np.quantile(s1s, np.array([0.01, 0.99])))
    print(np.quantile(s2s, np.array([0.01, 0.99])))
    for i in range(dim - 2):
        print(np.quantile(srest[...,i], np.array((0.01, 0.99))))

    # np.save("data/banana/X.npy", X, False)
    # np.save("data/banana/X-easy.npy", X, False)
