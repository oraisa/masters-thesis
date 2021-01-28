
import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.2,
    prop_sigma = np.repeat(0.007 * 5, 2),
    r_clip_bound = 5,
    ocu = False,
    grw = False
)
