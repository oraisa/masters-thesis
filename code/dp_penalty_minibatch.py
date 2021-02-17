import numpy as np
import numpy.random
import scipy.stats as stats
from scipy.special import binom
import numpy.linalg
import fourier_accountant as fa
import util

class MinibatchPenaltyParams:
    def __init__(self, tau, batch_size, r_clip_bound, prop_sigma, ocu, grw):
        self.tau = tau
        self.batch_size = batch_size
        self.clip_bound = r_clip_bound
        self.prop_sd = prop_sigma
        self.ocu = ocu
        self.grw = grw

def one_iter_RDP_epsilon(alpha, tau):
    return alpha / (2 * tau)

def total_RDP_epsilon(alpha, n, b, tau):
    gamma = b / n
    def calculate_exp_inner(j):
        return (
            j * np.log(gamma) + np.log(binom(alpha, j))
            + (j - 1) * one_iter_RDP_epsilon(j, tau) + np.log(2)
        )
    def calculate_exp(value):
        # For some values the np.exp overflows and is set to infinity
        # This is not a problem as then infinity is returned and 
        # it is handled as a large epsilon would be
        old_warnings = np.seterr(over="ignore")
        rval = np.exp(value)
        np.seterr(**old_warnings)
        return rval

    sum = np.sum(
        calculate_exp(calculate_exp_inner(j))
        for j in range(3, alpha)
    )
    exp_eps_2 = np.exp(one_iter_RDP_epsilon(2, tau))
    min_in_log = min(4 * (exp_eps_2 - 1), 2 * exp_eps_2)
    in_log = 1 + gamma**2 * binom(alpha, 2) * min_in_log + sum
    return 1 / (alpha - 1) * np.log(in_log)

def iterations_with_alpha(epsilon, delta, alpha, n, b, tau):
    eps_prime = total_RDP_epsilon(alpha, n, b, tau)
    return max(0, int((epsilon - np.log(1 / delta) / (alpha - 1)) / eps_prime))

def maximize_iterations(epsilon, delta, n, b, tau, max_alpha=200):
    max_iters = -1
    for alpha in range(2, max_alpha):
        iters = iterations_with_alpha(epsilon, delta, alpha, n, b, tau)
        if iters > max_iters:
            max_iters = iters
    return max(0, max_iters)

def adp_delta(iters, epsilon, tau, n, b):
    return fa.get_delta_S(target_eps=epsilon, sigma=np.sqrt(tau), q=b/n, ncomp=int(iters))

def adp_iters(epsilon, delta, tau, n, b):
    low_iters = 0
    up_iters = 1024
    while adp_delta(up_iters, epsilon, tau, n, b) < delta:
        up_iters *= 2
    while int(up_iters) - int(low_iters) > 1:
        new_iters = (low_iters + up_iters) / 2
        new_delta = adp_delta(new_iters, epsilon, tau, n, b)
        if new_delta > delta:
            up_iters = new_iters
        else:
            low_iters = new_iters

    if adp_delta(int(up_iters), epsilon, tau, n, b) < delta:
        return int(up_iters)
    else:
        return int(low_iters)

def d(current, proposal):
    return np.sqrt(np.sum((current - proposal)**2))

def c(current, proposal, n, b, temp_scale, clip_bound):
    return (n / b * temp_scale * 2 * clip_bound * d(current, proposal)
        + (1 - 1 / b) * (n * temp_scale / b)**2 * clip_bound**2 * d(current, proposal)**2
        + 2 / b * (n * temp_scale / b)**2 * (b - 1) * clip_bound**2 * d(current, proposal)**2
    )

def sigma(current, proposal, tau, n, b, temp_scale, clip_bound):
    return np.sqrt(tau) * c(current, proposal, n, b, temp_scale, clip_bound)

def dp_penalty_minibatch(problem, epsilon, delta, params, verbose=True, use_fa=True):
    one_component = params.ocu
    if params.grw:
        one_component = True # Guided random walk requires one component updates

    temp_scale = problem.temp_scale
    data = problem.data
    beta_0 = problem.theta0

    dim = beta_0.size
    n, data_dim = data.shape

    if use_fa:
        T = adp_iters(epsilon, delta, params.tau, n, params.batch_size)
    else:
        T = maximize_iterations(epsilon, delta, n, params.batch_size, params.tau)

    if verbose:
        print("Max iterations: {}".format(T))
    if params.grw:
        prop_dir = np.random.choice([-1, 1], dim)

    chain = np.zeros((T + 1, dim))
    chain[0] = beta_0
    sigmas = np.zeros(T + 1)
    clipped = np.zeros(T + 1)
    accepted = 0

    for i in range(T):
        current = chain[i]

        if one_component:
            update_component = np.random.randint(dim)
            proposal = current.copy()
            if params.grw:
                magnitude = np.abs(stats.norm.rvs(size=1, loc=0, scale=params.prop_sd[update_component]))
                proposal[update_component] += prop_dir[update_component] * magnitude
            else:
                proposal[update_component] += stats.norm.rvs(size=1, loc=0, scale=params.prop_sd)
        else:
            proposal = stats.norm.rvs(size=dim, loc=current, scale=params.prop_sd)


        X_batch_inds = np.random.choice(n, params.batch_size, replace=False)
        X_batch = data[X_batch_inds, :]

        lpc = problem.log_prior(current)
        lpp = problem.log_prior(proposal)
        llc = problem.log_likelihood_no_sum(current, X_batch)
        llp = problem.log_likelihood_no_sum(proposal, X_batch)
        ratio = llp - llc

        clip = params.clip_bound * d(current, proposal)
        clipped[i + 1] = np.sum(np.abs(ratio) > clip)
        ratio = np.clip(ratio, -clip, clip)

        R = ratio.mean()
        batch_var = (n * temp_scale)**2 / params.batch_size * ratio.var()
        # batch_vars[i + 1] = batch_var
        lambd = lpp - lpc + R * n * temp_scale
        s = sigma(current, proposal, params.tau, n, params.batch_size, temp_scale, params.clip_bound)
        sigmas[i + 1] = s
        d_hat = lambd + stats.norm.rvs(size=1, scale=s)

        random_val = stats.uniform.rvs(size=1)
        accept = np.exp(d_hat - (s**2 + batch_var) / 2) > random_val

        if accept:
            chain[i + 1] = proposal
            accepted += 1
        else:
            chain[i + 1] = current
            if params.grw:
                prop_dir[update_component] *= -1
        if verbose and (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))

    return util.MCMCResult(
        problem, chain, chain, T, accepted,
        np.sum(clipped) / params.batch_size / T, np.nan
    )
