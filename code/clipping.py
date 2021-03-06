import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import banana_model
import gauss_model
import mmd
import multiprocessing
import functools

class MCMCResult:
    def __init__(self, chain, clipped, clip_diff, orig_ratios, accepts, diff_accepts):
        self.chain = chain 
        self.final_chain = chain[int((chain.shape[0] - 1) / 2) + 1:, :]
        self.clipped = clipped
        self.clip_diff = clip_diff
        self.orig_ratios = orig_ratios
        self.accepts = accepts
        self.diff_accepts = diff_accepts

def rwmh(problem, iters, prop_sigma, clip_bound, theta0):
    data = problem.data
    dim = theta0.shape[0]
    chain = np.zeros((iters + 1, dim))
    chain[0] = theta0
    clipped = np.zeros(iters + 1)
    clip_diff = np.zeros(iters + 1)
    orig_ratios = np.zeros(iters + 1)
    accepts = 0
    diff_accepts = np.zeros(iters + 1)
    llc = problem.log_likelihood_no_sum(theta0, data)
    for i in range(iters):
        current = chain[i, :]
        prop = stats.norm.rvs(size=dim, loc=current, scale=prop_sigma)

        lpc = problem.log_prior(current)
        lpp = problem.log_prior(prop)
        llp = problem.log_likelihood_no_sum(prop, data)

        ratio = llp - llc
        orig_ratio = ratio
        orig_ratios[i + 1] = np.sum(orig_ratio)

        clip = clip_bound * np.sqrt(np.sum(current - prop)**2)
        clipped[i + 1] = np.sum(np.abs(ratio) > clip)
        ratio = np.clip(ratio, -clip, clip)
        clip_diff[i + 1] = np.sum(orig_ratio - ratio)

        lambd = np.sum(ratio) + lpp - lpc
        u = np.log(np.random.rand())
        accept = u < lambd
        orig_accept = u < np.sum(orig_ratio) + lpp - lpc
        if accept != orig_accept:
            diff_accepts[i + 1] = 1
        if accept:
            chain[i + 1, :] = prop
            accepts += 1
            llc = llp
        else:
            chain[i + 1, :] = current
        if (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))
    return MCMCResult(chain, clipped, clip_diff, orig_ratios, accepts, diff_accepts)

def hmc(problem, iters, eta, L, mass, clip_bound, theta0):
    data = problem.data
    dim = theta0.shape[0]
    chain = np.zeros((iters + 1, dim))
    chain[0] = theta0
    clipped = np.zeros(iters + 1)
    clip_diff = np.zeros(iters + 1)
    orig_ratios = np.zeros(iters + 1)
    accepts = 0
    diff_accepts = np.zeros(iters + 1)

    def grad_fun(theta):
        return np.sum(problem.log_likelihood_grads(theta, data), axis=0) + problem.log_prior_grad(theta)

    grad = grad_fun(theta0)
    llc = problem.log_likelihood_no_sum(theta0, data)
    for i in range(iters):
        current = chain[i, :]
        p = stats.norm.rvs(size=dim) * np.sqrt(mass)
        p_orig = p
        prop = current
        grad_new = grad

        for j in range(L):
            p += 0.5 * eta * grad_new
            prop += eta * p / mass
            grad_new = grad_fun(prop)
            p += 0.5 * eta * grad_new

        llp = problem.log_likelihood_no_sum(prop, data)
        r = llp - llc 
        r_orig = r
        clip = clip_bound * np.sqrt(np.sum(current - prop)**2)
        clipped[i + 1] = np.sum(np.abs(r) > clip)
        r = np.clip(r, -clip, clip)
        clip_diff[i + 1] = np.sum(r_orig - r)

        lpc = problem.log_prior(current)
        lpp = problem.log_prior(prop)

        d_kin = 0.5 * (np.sum(p_orig**2 / mass) - np.sum(p**2 / mass))
        dH = np.sum(r) + d_kin + lpp - lpc 
        dH_orig = np.sum(r_orig) + d_kin + lpp - lpc
        u = np.log(np.random.rand())
        accept = u < dH
        orig_accept = u < dH_orig
        if accept != orig_accept:
            diff_accepts[i + 1] = 1
        if accept:
            chain[i + 1, :] = prop
            accepts += 1
            llc = llp
            grad = grad_new
        else:
            chain[i + 1, :] = current
        if (i + 1) % 100 == 0:
            print("Iteration: {}".format(i + 1))
    return MCMCResult(chain, clipped, clip_diff, orig_ratios, accepts, diff_accepts)

dim = 2
n = 100000

def run_chain(init, algo, **args):
    return algo(theta0=init, problem=get_problem(), **args)
def get_problem():
    # return gauss_model.get_problem(dim=dim, n=n)
    return banana_model.get_problem(dim=dim, a=20, n0=1000, n=n)
if __name__ == "__main__":
    np.random.seed(43726482)

    problem = get_problem()

    clip_bound = 1000
    theta0 = np.zeros(dim)
    theta0[1] = 3
    inits = [theta0 + np.random.normal(scale=problem.true_posterior.std(axis=0), size=dim) for _ in range(4)]

    mass = np.ones(dim)

    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool() as pool:
        results = list(pool.map(functools.partial(
            run_chain, algo=hmc, iters=400, eta=0.001, L=10, mass=mass,
            clip_bound=clip_bound
        ), inits))
    result = results[0]
    total_chain = np.stack([result.final_chain for result in results])
    r_hat = mmd.r_hat(total_chain)
    print("R hat: {}".format(r_hat))

    iters = result.chain.shape[0] - 1
    print("Acceptance: {}".format(result.accepts / iters))
    print("Clipping: {}".format(np.sum(result.clipped) / iters / n))
    print("Different decisions: {}".format(np.sum(result.diff_accepts)))
    posterior = problem.true_posterior
    print("MMD: {}".format(mmd.mmd(result.final_chain, posterior)))
    # alt_posterior = banana.generate_posterior_samples(1000, data, 1)
    # print("Base MMD: {}".format(mmd.mmd(alt_posterior, posterior)))

    fig, axes = plt.subplots(dim)
    for res in results:
        for i in range(dim):
            axes[i].plot(res.chain[:, i])
    plt.show()

    # fig, axes = plt.subplots()
    # posterior = banana.generate_posterior_samples(10000, data, 1)
    # axes.scatter(posterior[:,0], posterior[:,1], alpha=0.1)
    # axes.scatter(result.final_chain[:, 0], result.final_chain[:, 1])
    # plt.show()

    # fig, axes = plt.subplots(1, 2)
    # inds = np.arange(iters + 1)[result.diff_accepts == 1]
    # for i in inds:
    #     axes[0].axvline(i, color="red")
    # axes[0].plot(result.clip_diff)

    # for i in inds:
    #     axes[1].axvline(i, color="red")
    # axes[1].plot(result.orig_ratios)
    # plt.show()
