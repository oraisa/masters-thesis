
import numpy as np
import hmc
params = hmc.HMCParams(
    tau = 0.08,
    tau_g = 0.20,
    L = 10,
    eta = 0.015,
    mass = 1,
    r_clip = 2.5,
    grad_clip = 3.5,
)
