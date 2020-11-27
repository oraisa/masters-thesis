import jax.numpy as np
import jax
import jax.scipy.stats as stats
# import scipy.stats as scipystats
import matplotlib.pyplot as plt
import numpy.random as npr

@jax.jit
def log_prior(theta, a, b, m, tau0):
    return (
        stats.norm.logpdf(theta[0], scale=1 / np.sqrt(tau0))
        + stats.norm.logpdf(
            theta[1] + a * (theta[0] - m)**2 + b, scale=1 / np.sqrt(tau0)
        )
        + np.sum(stats.norm.logpdf(theta[2:], scale=1 / np.sqrt(tau0)))
    )


@jax.jit 
def log_likelihood_per_sample(x, theta, a, b, m, sigma1, sigma2, sigma3, dim):
    logc1 = -np.log(sigma1 * np.sqrt(2 * np.pi))
    logc2 = -np.log(sigma2 * np.sqrt(2 * np.pi))
    logc3 = -np.log(sigma3 * np.sqrt(2 * np.pi))

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

log_likelihood_no_sum = jax.jit(jax.vmap(
    log_likelihood_per_sample, in_axes=(0, None, None, None, None, None, None, None, None)
))
log_likelihood = jax.jit(lambda X, theta: np.sum(log_likelihood_no_sum(X, theta)))
log_likelihood_grads = jax.jit(jax.vmap(
    jax.grad(log_likelihood_per_sample, 1), 
    in_axes=(0, None, None, None, None, None, None, None, None)
))
log_prior_grad = jax.jit(jax.grad(log_prior, 0))

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


class BananaModel:
    def __init__(self, dim=2):
        self.tau0 = 0.001
        self.tau1 = 0.05
        self.tau2 = 0.4
        self.tau3 = 10
        self.sigma1 = 1 / np.sqrt(self.tau1)
        self.sigma2 = 1 / np.sqrt(self.tau2)
        self.sigma3 = 1 / np.sqrt(self.tau3)
        # a = 80.0
        # a = 40
        self.a = 20
        self.b = 0.0
        self.m = 0.0
        self.dim = dim

    def generate_test_data(self, seed=81638):
        n = 100000
        theta1 = 0
        theta2 = 3
        theta_rest = np.zeros(self.dim - 2)

        key = jax.random.PRNGKey(43247)
        keys = jax.random.split(key, 3)

        x1s = jax.random.normal(keys[0], (n,1)) * self.sigma1 + theta1
        x2s = jax.random.normal(keys[1], (n,1)) * self.sigma2 + theta2 + self.a * (theta1 - self.m)**2 + self.b
        xrest = jax.random.normal(keys[2], (n, self.dim - 2)) * self.sigma3 + theta_rest
        return np.hstack((x1s, x2s, xrest))

    def log_likelihood_no_sum(self, data, theta):
        return log_likelihood_no_sum(
            data, theta, self.a, self.b, self.m, self.sigma1, self.sigma2, self.sigma3, self.dim
        )
    def log_prior(self, theta):
        return log_prior(theta, self.a, self.b, self.m, self.tau0)

    def banana_density(self, theta1, theta2, mu1, mu2, sigma1, sigma2, a, b, m):
        return (
            stats.norm.pdf(theta1, loc=mu1, scale=sigma1)
            * stats.norm.pdf(theta2 + a * (theta1 - m)**2 + b, loc=mu2, scale=sigma2)
        )

    def banana_g2(self, x1, x2, m):
        return x2 - self.a * (x1 - m)**2 - self.b

    def compute_posterior_params(self, X, T):
        n = X.shape[0]
        mu1 = (T * n * self.tau1 * X[..., 0].mean()) / (T * n * self.tau1 + self.tau0)
        mu2 = (T * n * self.tau2 * X[..., 1].mean()) / (T * n * self.tau2 + self.tau0)
        murest = (T * n * self.tau3 * X[...,2:].mean(axis=0)) / (T * n * self.tau3 + self.tau0)
        sigma1_p = 1 / np.sqrt(T * n * self.tau1 + self.tau0)
        sigma2_p = 1 / np.sqrt(T * n * self.tau2 + self.tau0)
        sigma3_p = 1 / np.sqrt(T * n * self.tau3 + self.tau0)
        return (mu1, mu2, murest, sigma1_p, sigma2_p, sigma3_p)

    def generate_posterior_samples(self, n, X, T, key=None):
        mu1, mu2, murest, sigma1_p, sigma2_p, sigma3_p = self.compute_posterior_params(X, T)

        if key is None:
            key = jax.random.PRNGKey(56437)
        keys = jax.random.split(key, 3)

        s1s = jax.random.normal(keys[0], (n,1)) * sigma1_p + mu1
        s2s = self.banana_g2(s1s, jax.random.normal(keys[1], (n,1)) * sigma2_p + mu2, self.m)
        srest = jax.random.normal(keys[2], (n, self.dim - 2)) * sigma3_p + murest
        return np.hstack((s1s, s2s, srest))

    def scatterplot_posterior(self, X, ax, T):
        post = self.generate_posterior_samples(1000, X, T)
        ax.scatter(post[:, 0], post[:, 1])

    def plot_posterior(self, X, ax, T):
        mu1, mu2, _, sigma1_p, sigma2_p, __ = self.compute_posterior_params(X, T)
        xs = np.linspace(-0.1, 0.1, 1000)
        ys = np.linspace(-0.1, 0.1, 1000) + mu2
        X, Y = np.meshgrid(xs, ys)
        Z = self.banana_density(X, Y, mu1, mu2, sigma1_p, sigma2_p, self.a, self.b, self.m)
        ax.contour(X, Y, Z)

if __name__ == "__main__":
    T = 100 / 100000
    npr.seed(463728)

    banana = BananaModel()
    X = banana.generate_test_data()

    fig, ax = plt.subplots()
    banana.scatterplot_posterior(X, ax, 1)
    banana.plot_posterior(X, ax, 1)
    plt.show()
    # plt.savefig("../latex/figures/banana-nontempered/posterior-easy.pdf")

    # fig, ax = plt.subplots()
    # scatterplot_posterior(X, ax, T)
    # plot_posterior(X, ax, T)
    # plt.show()
    # plt.savefig("../latex/figures/banana-tempered/posterior-easy.pdf")

    post = banana.generate_posterior_samples(10000, X, 1)
    print(np.quantile(post[:, 0], np.array([0.01, 0.99])))
    print(np.quantile(post[:, 1], np.array([0.01, 0.99])))
    for i in range(banana.dim - 2):
        print(np.quantile(post[:, i + 2], np.array((0.01, 0.99))))

    # np.save("data/banana/X.npy", X, False)
    # np.save("data/banana/X-easy.npy", X, False)