
import numpy as np
import hmc
params = hmc.HMCParams(
    tau = 0.05,
    tau_g = 0.10,
    L = 5,
    eta = 0.00025,
    mass = 1,#np.hstack((np.array((0.1, 1)), np.repeat(2, 28))),
    r_clip = 2.5,
    grad_clip = 6.0,
    # tau = 0.05,
    # tau_g = 0.10,
    # L = 5,
    # eta = 0.0002,
    # mass = np.hstack((np.array((0.1, 1)), np.repeat(2, 28))),
    # r_clip = 3.1,
    # grad_clip = 6.0,
)
