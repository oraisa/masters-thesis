
import numpy as np
import hmc
params = hmc.HMCParams(
        tau = 0.1,
        tau_g = 0.4,
        L = 10,
        eta = 0.0005,
        mass = 1,
        r_clip = 2,
        grad_clip = 1.0,
)
