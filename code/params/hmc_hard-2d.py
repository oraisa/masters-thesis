
import numpy as np
import hmc
params = hmc.HMCParams(
    tau = 0.15,
    tau_g = 0.25,
    L = 5,
    eta = 0.00045,
    mass = 1,
    r_clip = 3.5,
    grad_clip = 2.8,
)
