
import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.10,
    prop_sigma = np.repeat(0.0015, 10),
    r_clip_bound = 3,
    ocu = True,
    grw = True
)
