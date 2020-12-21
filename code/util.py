import jax 
import jax.numpy as np


class Problem:
    def __init__(self, log_likelihood_per_sample, log_prior, data):
        """Log likelihood per sample should have signature (theta, sample) -> float"""
        self.log_likelihood_per_sample = log_likelihood_per_sample
        self.log_prior = log_prior
        self.data = data

        self.log_likelihood_no_sum = jax.jit(jax.vmap(self.log_likelihood_per_sample, in_axes=(None, 0)))
        self.log_likelihood = jax.jit(lambda theta, X: np.sum(self.log_likelihood_no_sum(theta, X)))
        self.log_likelihood_grads = jax.jit(
            jax.vmap(jax.grad(self.log_likelihood_per_sample, 0), in_axes=(None, 0))
        )
        self.log_prior_grad = jax.jit(jax.grad(self.log_prior))
        self.log_likelihood_grad_clipped = clip_grad_fun(self.log_likelihood_grads)

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
