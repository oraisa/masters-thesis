
import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.15,
    prop_sigma = np.array((0.008, 0.007)) * 1,
    r_clip_bound = 3,
    ocu = True,
    grw = True
)