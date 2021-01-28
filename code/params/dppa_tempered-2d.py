
import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.2,
    prop_sigma = np.repeat(0.008 * 5, 2),
    r_clip_bound = 3.5,
    ocu = True,
    grw = True
)
