
import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.06,
    prop_sigma = np.hstack((np.array((20, 7)), np.repeat(5, 28))) * 0.00014,
    r_clip_bound = 3,
    ocu = False,
    grw = False
)
