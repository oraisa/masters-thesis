import numpy as np
import scipy.stats as stats
import scipy.special as spec
import util

class HMCParams:
    def __init__(self, tau, tau_g, L, eta, mass, r_clip, grad_clip):
        self.tau = tau 
        self.tau_g = tau_g 
        self.L = L
        self.eta = eta 
        self.mass = mass 
        self.r_clip = r_clip 
        self.grad_clip = grad_clip

class GradClipCounter:
    def __init__(self):
        self.clipped_grad = 0
        self.grad_accesses = 0

def zcdp_iters(epsilon, delta, params, n, compute_less_grad=False):
    rho = (np.sqrt(epsilon - np.log(delta)) - np.sqrt(-np.log(delta)))**2
    rho_l = 1 / (2 * params.tau**2 * n)
    rho_g = 1 / (2 * params.tau_g**2 * n)
    # print("rho_l: {}".format(rho_l))
    # print("rho_g: {}".format(rho_g))

    if compute_less_grad:
        iters = int((rho - rho_g) / (rho_l + params.L * rho_g))
    else:
        iters = int(rho / (rho_l + (params.L + 1) * rho_g))
    return iters

def adp_delta(k, epsilon, params, n, compute_less_grad=False):
    tau_l = params.tau
    tau_g = params.tau_g
    L = params.L
    grad_evals = k * L + 1 if compute_less_grad else k * (L + 1)
    mu = k / (2 * tau_l**2 * n) + grad_evals / (2 * tau_g**2 * n)
    term1 = spec.erfc((epsilon - mu) / (2 * np.sqrt(mu)))
    term2 = np.exp(epsilon) * spec.erfc((epsilon + mu) / (2 * np.sqrt(mu)))
    return (0.5 * (term1 - term2)).sum()

def adp_iters(epsilon, delta, params, n, compute_less_grad=False):
    low_iters = zcdp_iters(epsilon, delta, params, n, compute_less_grad)
    up_iters = max(low_iters, 1)
    while adp_delta(up_iters, epsilon, params, n, compute_less_grad) < delta:
        up_iters *= 2
    while int(up_iters) - int(low_iters) > 1:
        new_iters = (low_iters + up_iters) / 2
        new_delta = adp_delta(new_iters, epsilon, params, n, compute_less_grad)
        if new_delta > delta:
            up_iters = new_iters
        else:
            low_iters = new_iters

    if adp_delta(int(up_iters), epsilon, params, n, compute_less_grad) < delta:
        return int(up_iters)
    else:
        return int(low_iters)

def hmc(problem, theta0, epsilon, delta, params, verbose=True, use_adp=True, compute_less_grad=False):

    data = problem.data
    n, data_dim = data.shape
    dim = theta0.size
    temp_scale = problem.temp_scale

    tau = params.tau 
    tau_g = params.tau_g
    L = params.L
    eta = params.eta
    mass = params.mass
    r_clip = params.r_clip
    grad_clip = params.grad_clip

    if not use_adp:
        iters = zcdp_iters(epsilon, delta, params, n, compute_less_grad)
    else:
        iters = adp_iters(epsilon, delta, params, n, compute_less_grad)

    if verbose:
        print("Iterations: {}".format(iters))
    sigma = tau * np.sqrt(n)

    chain = np.zeros((iters + 1, dim))
    chain[0, :] = theta0
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
        return temp_scale * (ll_grad + stats.norm.rvs(size=dim, scale=grad_noise_sigma)) + pri_grad

    if compute_less_grad:
        grad = grad_fun(theta0)
    llc = problem.log_likelihood_no_sum(theta0, data)
    for i in range(iters):
        current = chain[i, :]
        #TODO: this assumes diagonal M
        p = stats.norm.rvs(size=dim) * np.sqrt(mass)
        p_orig = p.copy()
        prop = current.copy()
        if compute_less_grad:
            grad_new = grad.copy()
        else:
            grad_new = grad_fun(current)

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
        dp = 0.5 * np.sum(p_orig**2 / mass) - 0.5 * np.sum(p**2 / mass)
        dH = dp + temp_scale * (np.sum(r) + s) + lpp - lpc
        u = np.log(np.random.rand())

        if u < dH - 0.5 * (temp_scale * sigma * d * 2 * r_clip)**2:
            chain[i + 1, :] = prop
            if compute_less_grad:
                grad = grad_new
            llc = llp 
            accepts += 1
        else:
            chain[i + 1, :] = current
        if verbose and (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))

    if verbose:
        print("Gradient evals: {}".format(clipped_grad_counter.grad_accesses))
    return util.MCMCResult(
        problem, chain, leapfrog_chain, iters, accepts, np.sum(clipped_r) / n / iters,
        np.sum(clipped_grad_counter.clipped_grad) / n / clipped_grad_counter.grad_accesses
    )
    # return (
    #     chain, leapfrog_chain, accepts, clipped_r, iters,
    #     clipped_grad_counter.clipped_grad, clipped_grad_counter.grad_accesses
    # )
