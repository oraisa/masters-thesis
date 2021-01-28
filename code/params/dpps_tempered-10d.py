
import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.18,
    prop_sigma = np.repeat(0.02, 10),
    r_clip_bound = 3,
    ocu = False,
    grw = False
)
