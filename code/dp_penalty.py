import numpy as np
import scipy.stats as stats
import scipy.special as spec

class PenaltyParams:
    def __init__(self, tau, theta0, r_clip_bound, prop_sigma):
        self.tau = tau
        self.theta0 = theta0
        self.r_clip_bound = r_clip_bound
        self.prop_sigma = prop_sigma

def zcdp_iters(epsilon, delta, tau, n):
    rho = (np.sqrt(epsilon - np.log(delta)) - np.sqrt(-np.log(delta)))**2
    iters = int(2 * tau**2 * n * rho)
    return iters

def adp_delta(k, epsilon, tau, n):
    mu = 1 / (2 * tau**2 * n)
    # term1 = jspec.erfc(tau * jnp.sqrt(n) * (epsilon - k * mu) / (jnp.sqrt(2 * k)))
    # term2 = jnp.exp(epsilon) * jspec.erfc(tau * jnp.sqrt(n) * (epsilon + k * mu) / (jnp.sqrt(2 * k)))
    term1 = spec.erfc((epsilon - k * mu) / (2 * np.sqrt(mu * k)))
    term2 = np.exp(epsilon) * spec.erfc((epsilon + k * mu) / (2 * np.sqrt(mu * k)))
    return (0.5 * (term1 - term2)).sum()

def adp_iters(epsilon, delta, tau, n):
    low_iters = zcdp_iters(epsilon, delta, tau, n)
    up_iters = low_iters
    while adp_delta(up_iters, epsilon, tau, n) < delta:
        up_iters *= 2
    while int(up_iters) > int(low_iters):
        new_iters = (low_iters + up_iters) / 2
        new_delta = adp_delta(new_iters, epsilon, tau, n)
        if new_delta > delta:
            up_iters = new_iters
        else:
            low_iters = new_iters

    return int(low_iters)

def dp_penalty(problem, epsilon, delta, params, use_adp=True):
    dim = params.theta0.size
    data = problem.data
    n, data_dim = data.shape
    tau = params.tau
    theta0 = params.theta0
    r_clip_bound = params.r_clip_bound
    prop_sigma = params.prop_sigma

    if use_adp:
        iters = adp_iters(epsilon, delta, tau, n)
    else:
        iters = zcdp_iters(epsilon, delta, tau, n)
    print("Iterations: {}".format(iters))

    sigma = tau * np.sqrt(n)

    chain = np.zeros((iters + 1, dim))
    chain[0, :] = theta0
    clipped_r = np.zeros(iters)
    accepts = 0

    llc = problem.log_likelihood_no_sum(theta0, data)
    for i in range(iters):
        current = chain[i, :]
        prop = stats.norm.rvs(size=dim, loc=current, scale=prop_sigma)

        llp = problem.log_likelihood_no_sum(prop, data)
        r = llp - llc
        d = np.sqrt(np.sum((current - prop)**2))
        clip = d * r_clip_bound
        clipped_r[i] = np.sum(np.abs(r) > clip)
        r = np.clip(r, -clip, clip)

        lpp = problem.log_prior(prop)
        lpc = problem.log_prior(current)

        s = stats.norm.rvs(size=1, scale=sigma * d * 2 * r_clip_bound)
        lambd = np.sum(r) + lpp - lpc + s
        u = np.log(np.random.rand())

        if u < lambd - 0.5 * (sigma * d * 2 * r_clip_bound)**2:
            chain[i + 1, :] = prop
            llc = llp
            accepts += 1
        else:
            chain[i + 1, :] = current
        if (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))

    return chain, accepts, clipped_r, iters
