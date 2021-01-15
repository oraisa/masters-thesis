
import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
        tau = 0.08,
        prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.0005,
        r_clip_bound = 3,
        ocu = True,
        grw = True
)
