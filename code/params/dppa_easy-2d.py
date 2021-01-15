import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.1,
    prop_sigma = np.array((0.008, 0.007)) * 2,
    r_clip_bound = 1.8,
    ocu = True,
    grw = True
)
