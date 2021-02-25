
import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.1,
    prop_sigma = np.repeat(0.0015, 2),
    r_clip_bound = 8.5,
    ocu = True,
    grw = True
)
