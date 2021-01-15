import numpy as np
import scipy.stats as stats
import scipy.special as spec
import util

class PenaltyParams:
    def __init__(self, tau, r_clip_bound, prop_sigma, ocu, grw):
        self.tau = tau
        self.r_clip_bound = r_clip_bound
        self.prop_sigma = prop_sigma
        self.ocu = ocu
        self.grw = grw

def zcdp_iters(epsilon, delta, tau, n):
    rho = (np.sqrt(epsilon - np.log(delta)) - np.sqrt(-np.log(delta)))**2
    iters = int(2 * tau**2 * n * rho)
    return iters

def adp_delta(k, epsilon, tau, n):
    mu = 1 / (2 * tau**2 * n)
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

def dp_penalty(problem, epsilon, delta, params, verbose=True, use_adp=True):
    ocu = params.ocu
    if params.grw:
        ocu = True # GRW requires one component updates

    dim = problem.theta0.size
    data = problem.data
    n, data_dim = data.shape
    temp_scale = problem.temp_scale

    tau = params.tau
    theta0 = problem.theta0
    r_clip_bound = params.r_clip_bound
    prop_sigma = params.prop_sigma

    if use_adp:
        iters = adp_iters(epsilon, delta, tau, n)
    else:
        iters = zcdp_iters(epsilon, delta, tau, n)

    if verbose:
        print("Iterations: {}".format(iters))

    if params.grw:
        prop_dir = np.random.choice([-1, 1], dim)

    sigma = tau * np.sqrt(n)

    chain = np.zeros((iters + 1, dim))
    chain[0, :] = theta0
    clipped_r = np.zeros(iters)
    accepts = 0

    llc = problem.log_likelihood_no_sum(theta0, data)
    for i in range(iters):
        current = chain[i, :]

        if ocu:
            update_component = np.random.randint(dim)
            prop = current.copy()
            if params.grw:
                magnitude = np.abs(stats.norm.rvs(
                    size=1, loc=0, scale=params.prop_sigma[update_component]
                ))
                prop[update_component] += prop_dir[update_component] * magnitude
            else:
                prop[update_component] += stats.norm.rvs(
                    size=1, loc=0, scale=params.prop_sigma[update_component]
                )
        else:
            prop = stats.norm.rvs(size=dim, loc=current, scale=params.prop_sigma)

        llp = problem.log_likelihood_no_sum(prop, data)
        r = llp - llc
        d = np.sqrt(np.sum((current - prop)**2))
        clip = d * r_clip_bound
        clipped_r[i] = np.sum(np.abs(r) > clip)
        r = np.clip(r, -clip, clip)

        lpp = problem.log_prior(prop)
        lpc = problem.log_prior(current)

        s = stats.norm.rvs(size=1, scale=sigma * d * 2 * r_clip_bound)
        lambd = temp_scale * (np.sum(r) + s) + lpp - lpc
        u = np.log(np.random.rand())

        if u < lambd - 0.5 * (temp_scale * sigma * d * 2 * r_clip_bound)**2:
            chain[i + 1, :] = prop
            llc = llp
            accepts += 1
        else:
            chain[i + 1, :] = current
            if params.grw:
                prop_dir[update_component] *= -1
        if verbose and (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))

    return util.MCMCResult(
        problem, chain, chain, iters, accepts, np.sum(clipped_r) / n / iters, np.nan
    )
