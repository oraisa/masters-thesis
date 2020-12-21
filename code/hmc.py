import numpy as np
import scipy.stats as stats 
import banana_model as banana
import arviz 
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mmd
import util

class HMCParams:
    def __init__(self, tau, tau_g, L, eta, mass, r_clip, grad_clip, theta0):
        self.tau = tau 
        self.tau_g = tau_g 
        self.L = L
        self.eta = eta 
        self.mass = mass 
        self.r_clip = r_clip 
        self.grad_clip = grad_clip
        self.theta0 = theta0

class GradClipCounter:
    def __init__(self):
        self.clipped_grad = 0
        self.grad_accesses = 0

def hmc(problem, epsilon, delta, params):

    data = problem.data
    n, data_dim = data.shape
    dim = params.theta0.size

    tau = params.tau 
    tau_g = params.tau_g
    L = params.L
    eta = params.eta
    mass = params.mass
    r_clip = params.r_clip
    grad_clip = params.grad_clip

    rho = (np.sqrt(epsilon - np.log(delta)) - np.sqrt(-np.log(delta)))**2
    rho_l = 1 / (2 * tau**2 * n)
    rho_g = (L + 1) / (2 * tau_g**2 * n)
    print("rho_l: {}".format(rho_l))
    print("rho_g: {}".format(rho_g))

    iters = int(rho / (rho_l + rho_g))
    print("Iterations: {}".format(iters))

    sigma = tau * np.sqrt(n)

    chain = np.zeros((iters + 1, dim))
    chain[0, :] = params.theta0
    leapfrog_chain = np.zeros((iters * L, dim))
    clipped_r = np.zeros(iters)
    clipped_grad_counter = GradClipCounter()
    accepts = 0

    grad_noise_sigma = 2 * tau_g * np.sqrt(n) * grad_clip

    def grad_fun(theta):
        ll_grad, clips = problem.log_likelihood_grad_clipped(grad_clip, theta, data)
        clipped_grad_counter.clipped_grad += clips
        clipped_grad_counter.grad_accesses += 1

        pri_grad = problem.log_prior_grad(theta)
        return ll_grad + pri_grad + stats.norm.rvs(size=dim, scale=grad_noise_sigma)

    grad = grad_fun(params.theta0)
    llc = problem.log_likelihood_no_sum(params.theta0, data)
    for i in range(iters):
        current = chain[i, :]
        #TODO: this assumes diagonal M
        p = stats.norm.rvs(size=dim) * np.sqrt(mass)
        p_orig = p.copy()
        prop = current.copy()
        grad_new = grad.copy()

        for j in range(L):
            p += 0.5 * eta * (grad_new)# - 0.5 * grad_noise_sigma**2 * p / mass)
            prop += eta * p / mass
            leapfrog_chain[i * L + j] = prop
            grad_new = grad_fun(prop)
            p += 0.5 * eta * (grad_new)# - 0.5 * grad_noise_sigma**2 * p / mass)

        llp = problem.log_likelihood_no_sum(prop, data)
        r = llp - llc
        d = np.sqrt(np.sum((current - prop)**2))
        clip = d * r_clip
        clipped_r[i] = np.sum(np.abs(r) > clip)
        r = np.clip(r, -clip, clip)

        lpp = problem.log_prior(prop)
        lpc = problem.log_prior(current)

        s = stats.norm.rvs(size=1, scale=sigma * d * 2 * r_clip)
        dH = 0.5 * np.sum(p_orig**2 / mass) - 0.5 * np.sum(p**2 / mass) + np.sum(r) + lpp - lpc + s
        u = np.log(np.random.rand())

        if u < dH - 0.5 * (sigma * d * 2 * r_clip)**2:
            chain[i + 1, :] = prop 
            grad = grad_new 
            llc = llp 
            accepts += 1
        else:
            chain[i + 1, :] = current
        if (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))

    return (
        chain, leapfrog_chain, accepts, clipped_r, iters,
        clipped_grad_counter.clipped_grad, clipped_grad_counter.grad_accesses
    )


# lines = plt.plot(leapfrog_chain[:, 0], leapfrog_chain[:, 1])
# plt.show()

# mcmc_trace = arviz.from_netcdf("abalone_post.nc")
# mcmc_samples = np.concatenate(mcmc_trace.posterior.theta.values, axis=0)
# fig, axes = plt.subplots(dim)
# for i in range(dim):
#     sns.kdeplot(mcmc_samples[:, i], ax=axes[i])
#     sns.kdeplot(final_chain[:, i], ax=axes[i])
# plt.show()

