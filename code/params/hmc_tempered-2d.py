
import numpy as np
import hmc
params = hmc.HMCParams(
    tau = 0.2,
    tau_g = 0.6,
    L = 10,
    eta = 0.01,
    mass = 1,
    r_clip = 2.5,
    grad_clip = 2.0,
)
