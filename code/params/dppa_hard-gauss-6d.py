import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.07,
    prop_sigma = np.repeat(0.0002, 6),
    r_clip_bound = 25,
    ocu = True,
    grw = True
)
