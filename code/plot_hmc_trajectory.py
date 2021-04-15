import numpy as np 
import matplotlib.pyplot as plt
import circle
import hmc

np.random.seed(46327)

problem = circle.problem()
n, data_dim = problem.data.shape
dim = 2

epsilon = 1
delta = 0.1 / n

params = hmc.HMCParams(
    tau = 0.6,
    tau_g = 1.9,
    L = 4 * 12,
    eta = 0.07,
    mass = np.array((1, 1)),
    r_clip = 0.001,
    grad_clip = 0.0015,
)

res = hmc.hmc(problem, np.array((0, 1)), epsilon, delta, params)

samples = res.chain
n_samples = samples.shape[0]
proposals = res.leapfrog_chain
props_per_sample = int(proposals.shape[0] / (samples.shape[0] - 1))
points = np.zeros((props_per_sample + 1, samples.shape[0], dim))
for i in range(samples.shape[0]):
    points[0, i, :] = samples[i, :]
    if i < samples.shape[0] - 1:
        points[1:, i, :] = proposals[
            (i * props_per_sample):((i+1) * props_per_sample),
            :
        ]

def plot_anim_frame(ind, ax):
    sample_i, prop_i, = np.unravel_index(ind, (n_samples, props_per_sample + 1))
    # ax.set_xlim(-4, 4)
    # ax.set_ylim(-4, 4)
    ax.plot(
        points[0:prop_i, sample_i, 0],
        points[0:prop_i, sample_i, 1]
    )
    ax.plot(
        points[0, sample_i, 0],
        points[0, sample_i, 1],
        linestyle='', marker='.'
    )
    circle.plot_density(problem, ax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_frames(start_i):
    fig, ax = plt.subplots(4, 3, figsize=(2.5 * 3, 2.5 * 4))
    for i in range(4):
        for j in range(3):
            ind = start_i + (3 * i + j) * 4
            plot_anim_frame(ind, ax[i, j])
            ax[i, j].set_title(
                "j = {}".format(ind % (props_per_sample + 1) + 1),
                # fontdict={"fontsize": "small"}
            )

plot_frames(304 + 48 * 10)
# plt.show()
plt.savefig("../Thesis/figures/hmc_trajectory.pdf")
