import numpy as np
import hmc

params = hmc.HMCParams(
    tau = 0.05,
    tau_g = 0.20,
    L = 5,
    eta = 0.00004,
    mass = 1,
    r_clip = 20,
    grad_clip = 25.0,
)
