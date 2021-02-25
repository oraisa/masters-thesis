import jax 
import jax.numpy as np
import scipy.optimize as optim
import mmd

class Problem:
    def __init__(
            self, log_likelihood_per_sample, log_prior, data, temp_scale,
            theta0, gen_true_posterior, plot_density_func, true_mean=None
    ):
        """Log likelihood per sample should have signature (theta, sample) -> float"""
        self.log_likelihood_per_sample = log_likelihood_per_sample
        self.log_prior = log_prior
        self.data = data
        self.temp_scale = temp_scale
        if gen_true_posterior is not None:
            self.gen_true_posterior = lambda n, key=None: gen_true_posterior(
                n, data, temp_scale, key
            )
            self.true_posterior = gen_true_posterior(1000, data, temp_scale)
        else:
            self.gen_true_posterior = None
            self.true_posterior = None
        self.true_mean = true_mean if true_mean is not None else np.mean(self.true_posterior, axis=0)
        self.theta0 = theta0
        self.dim = theta0.size
        self.plot_density_func = plot_density_func

        self.log_likelihood_no_sum = jax.jit(jax.vmap(self.log_likelihood_per_sample, in_axes=(None, 0)))
        self.log_likelihood = jax.jit(lambda theta, X: np.sum(self.log_likelihood_no_sum(theta, X)))

        self.log_likelihood_grads = jax.jit(
            jax.vmap(jax.grad(self.log_likelihood_per_sample, 0), in_axes=(None, 0))
        )
        self.log_likelihood_grad_clipped = clip_grad_fun(self.log_likelihood_grads)

        self.log_prior_grad = jax.jit(jax.grad(self.log_prior))

    def plot_density(self, ax):
        self.plot_density_func(self, ax)

    def get_start_point(self, i):
        key = jax.random.PRNGKey(4236482 + i)
        if self.true_posterior is not None:
            # inters = np.quantile(self.true_posterior, (0.25, 0.75), axis=0)
            # mul = (inters[1,:] - inters[0,:]).mean() * 0.5
            # print(mul)
            mul = self.true_posterior.std(axis=0).mean() * 1
            # ind = jax.random.choice(key, self.true_posterior.shape[0])
            # return self.true_posterior[ind, :]
            # print(mul)
        else:
            mul = 1
            # std_noise = jax.random.normal(key, shape=(self.dim,))
            # return self.theta0 + std_noise * 0.3
        return self.theta0 + jax.random.normal(key, shape=(self.dim,)) * mul

def clip_grad_fun(grad_fun):
    def return_fun(clip, *args):
        grads = grad_fun(*args)
        grads, did_clip = jax.vmap(clip_norm, in_axes=(0, None))(grads, clip)
        clipped_grad = np.sum(did_clip)
        return (np.sum(grads, axis=0), clipped_grad)
    return jax.jit(return_fun)

@jax.jit
def clip_norm(x, bound):
    norm = np.sqrt(np.sum(x**2))
    clipped_norm = np.min(np.array((norm, bound)))
    return (x / norm * clipped_norm, norm > bound)

class MCMCResult:
    def __init__(self, problem, chain, leapfrog_chain, iters, accepts, clipped_r, clipped_grad):
        self.chain = chain
        self.leapfrog_chain = leapfrog_chain
        self.clipped_r = clipped_r
        self.clipped_grad = clipped_grad
        posterior = problem.true_posterior

        if iters > 1:
            self.final_chain = chain[int((iters - 1) / 2) + 1:, :]
            self.acceptance = accepts / iters
            if posterior is not None:
                self.mean_error = mmd.mean_error(self.final_chain, posterior)
                self.cov_error = mmd.cov_error(self.final_chain, posterior)
                self.mmd = mmd.mmd(self.final_chain, posterior)
            else:
                if problem.true_mean is not None:
                    self.mean_error = np.sqrt(
                        np.sum(
                            (np.mean(self.final_chain, axis=0)
                             - problem.true_mean)**2
                        )
                    )
                else:
                    self.mean_error = np.nan
                self.cov_error = np.nan
                self.mmd = np.nan
        else:
            self.acceptance = np.nan
            self.mean_error = np.nan
            self.cov_error = np.nan
            self.mmd = np.nan
