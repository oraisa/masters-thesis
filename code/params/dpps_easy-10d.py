
import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
        tau = 0.08,
        prop_sigma = np.array((0.008, 0.007, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001)) * 0.8,
        r_clip_bound = 2,
        ocu = False,
        grw = False
)
