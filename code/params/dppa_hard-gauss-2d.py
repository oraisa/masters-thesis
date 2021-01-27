import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.1,
    prop_sigma = np.repeat(0.0002, 2),
    r_clip_bound = 45,
    ocu = True,
    grw = True
)
