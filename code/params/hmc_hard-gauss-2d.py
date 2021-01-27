
import numpy as np
import hmc

params = hmc.HMCParams(
    tau = 0.05,
    tau_g = 0.20,
    L = 8,
    eta = 0.00007,
    mass = 1,
    r_clip = 30,
    grad_clip = 29.0,
)
