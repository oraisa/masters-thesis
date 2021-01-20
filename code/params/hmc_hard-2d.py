
import numpy as np
import hmc
params = hmc.HMCParams(
    tau = 0.08,
    tau_g = 0.35,
    L = 20,
    eta = 0.00030,
    mass = np.array((0.5, 1)),
    r_clip = 2.1,
    grad_clip = 2.5,
    # tau = 0.1,
    # tau_g = 0.40,
    # L = 10,
    # eta = 0.00030,
    # mass = np.array((0.1, 1)),
    # r_clip = 2.1,
    # grad_clip = 1.5,
)
