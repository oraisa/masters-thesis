import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import functools
import dp_penalty
import dp_penalty_minibatch
import hmc
import params

if __name__ == "__main__":
    dpps_params = params.__dict__["dpps_easy-2d"].params
    mdpps_params = params.__dict__["mdpps_easy-2d"].params
    hmc_params = params.__dict__["hmc_easy-2d"].params

    n = 1e5
    delta = 0.1 / n
    epsilons = np.linspace(0.5, 6, 50)

    zcdp_iters = np.array([dp_penalty.zcdp_iters(eps, delta, dpps_params.tau, n) for eps in epsilons])
    adp_iters = np.array([dp_penalty.adp_iters(eps, delta, dpps_params.tau, n) for eps in epsilons])

    hmc_zcdp_iters = np.array([hmc.zcdp_iters(eps, delta, hmc_params, n) for eps in epsilons])
    hmc_adp_iters = np.array([hmc.adp_iters(eps, delta, hmc_params, n) for eps in epsilons])

    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool() as pool:
        mini_rdp_iters = np.array(list(pool.map(functools.partial(
            dp_penalty_minibatch.maximize_iterations, delta=delta, n=n,
            b=mdpps_params.batch_size, tau=mdpps_params.tau
            ), epsilons)
        ))
        mini_adp_iters = np.array(list(pool.map(functools.partial(
            dp_penalty_minibatch.adp_iters, delta=delta, n=n,
            b=mdpps_params.batch_size, tau=mdpps_params.tau
            ), epsilons)
        ))
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].set_title("DP penalty")
    ax[0].set_xlabel("Epsilon")
    ax[0].set_ylabel("Iterations")
    ax[0].plot(epsilons, zcdp_iters, label="zCDP")
    ax[0].plot(epsilons, adp_iters, label="ADP")
    ax[0].legend()

    ax[1].set_title("Minibatch DP penalty")
    ax[1].set_xlabel("Epsilon")
    ax[1].set_ylabel("Iterations")
    ax[1].plot(epsilons, mini_rdp_iters, label="RDP")
    ax[1].plot(epsilons, mini_adp_iters, label="ADP")
    ax[1].legend()

    ax[2].set_title("DP HMC")
    ax[2].set_xlabel("Epsilon")
    ax[2].set_ylabel("Iterations")
    ax[2].plot(epsilons, hmc_zcdp_iters, label="zCDP")
    ax[2].plot(epsilons, hmc_adp_iters, label="ADP")
    ax[2].legend()

    ax[0].set_ylim(0, 3000)
    ax[1].set_ylim(0, 3000)
    ax[2].set_ylim(0, 3000)

    plt.tight_layout()
    plt.savefig("../Thesis/figures/accountant_comparison.pdf")
