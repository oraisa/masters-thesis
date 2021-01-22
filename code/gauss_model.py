
import jax.numpy as np
import jax.scipy.stats as stats
import jax.scipy.linalg as linalg
import jax
import util
import matplotlib.pyplot as plt

@jax.jit
def log_prior(theta, sigma0):
    return np.sum(stats.norm.logpdf(theta, scale=sigma0))

@jax.jit
def log_likelihood_per_sample(theta, x, cov):
    return stats.multivariate_normal.logpdf(theta, mean=x, cov=cov)

@jax.jit
def log_likelihood_per_sample_fast(theta, x, chol):
    dim = theta.size
    # const = -0.5 * dim * np.log(2 * np.pi)
    # logdet = -np.sum(np.log(chol.diagonal()))
    z = linalg.solve_triangular(chol, x - theta, lower=True)
    return -0.5 * np.sum(z**2) #+ const + logdet

class GaussModel:
    def __init__(self, dim):
        self.dim = dim
        self.sigma0 = 100
        self.tau0 = 1 / self.sigma0**2
        self.cov = np.eye(dim)
        for i in range(int(self.dim / 2)):
            self.cov = jax.ops.index_update(
                self.cov, jax.ops.index[2*i:2*(i+1), 2*i:2*(i+1)],
                np.array(((1, 0.99), (0.99, 1))) / (i+1)
            )

        self.L = np.linalg.cholesky(self.cov)
        self.true_mean = np.hstack((np.array((0, 3)), np.zeros(dim - 2)))

    def log_prior(self, theta):
        return log_prior(theta, self.sigma0)

    def log_likelihood_per_sample(self, theta, x):
        return log_likelihood_per_sample_fast(theta, x, self.L)

    def generate_data(self, n):
        key = jax.random.PRNGKey(46283648)
        return jax.random.multivariate_normal(key, self.true_mean, self.cov, (n,))

    def generate_true_posterior(self, samples, data):
        n, d = data.shape
        key = jax.random.PRNGKey(8646546)

        sigma0_mat = np.eye(self.dim) * self.sigma0
        term1 = sigma0_mat @ np.linalg.inv(sigma0_mat + self.cov / n)
        mu_post = term1 @ np.mean(data, axis=0).reshape((-1, 1))
        cov_post = term1 @ self.cov / n
        return jax.random.multivariate_normal(key, mu_post.reshape(-1,), cov_post, (samples,))

def get_problem(dim, n):
    model = GaussModel(dim)
    data = model.generate_data(n)
    problem = util.Problem(
        model.log_likelihood_per_sample, model.log_prior,
        data, 1, model.true_mean, model.generate_true_posterior(1000, data)
    )
    return problem

if __name__ == "__main__":
    dim = 6
    model = GaussModel(dim)
    data = model.generate_data(100000)
    posterior = model.generate_true_posterior(1000, data)
    plt.scatter(posterior[:, 0], posterior[:, 1])
    plt.show()
    for i in range(dim):
        print(np.quantile(posterior[:, i], (0.01, 0.99)))

    # x = np.repeat(1, dim)
    # theta = np.repeat(2, dim)
    # print(log_likelihood_per_sample(x, theta, model.cov))
    # L = np.linalg.cholesky(model.cov)
    # print(log_likelihood_per_sample_fast(x, theta, L))

    # log_likelihood_per_sample(np.zeros(dim), np.zeros(dim), np.eye(dim))
    # log_likelihood_per_sample_fast(np.zeros(dim), np.zeros(dim), np.eye(dim))
    # from timeit import timeit
    # print(timeit(lambda: log_likelihood_per_sample(x, theta, model.cov), number=1))
    # print(timeit(lambda: log_likelihood_per_sample_fast(x, theta, L), number=1))