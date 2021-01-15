
import numpy as np
import hmc
params = hmc.HMCParams(
        tau = 0.05,
        tau_g = 0.2,
        L = 10,
        eta = 0.0002,
        mass = np.array((0.3, 1, 2, 2, 2, 2, 2, 2, 2, 2)),
        r_clip = 3,
        grad_clip = 3.0,
)
