import numpy as np
import hmc

params = hmc.HMCParams(
    tau = 0.6,
    tau_g = 1.9,
    L = 40,
    eta = 0.07,
    mass = np.array((1, 1)),
    r_clip = 0.001,
    grad_clip = 0.0015,
)
