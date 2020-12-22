import numpy as np
import scipy.stats as stats

class PenaltyParams:
    def __init__(self, tau, theta0, r_clip_bound, prop_sigma):
        self.tau = tau
        self.theta0 = theta0
        self.r_clip_bound = r_clip_bound
        self.prop_sigma = prop_sigma

def dp_penalty(problem, epsilon, delta, params):
    dim = params.theta0.size
    data = problem.data
    n, data_dim = data.shape
    tau = params.tau
    theta0 = params.theta0
    r_clip_bound = params.r_clip_bound
    prop_sigma = params.prop_sigma

    rho = (np.sqrt(epsilon - np.log(delta)) - np.sqrt(-np.log(delta)))**2
    iters = int(2 * tau**2 * n * rho)
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
