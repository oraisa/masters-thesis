import numpy as np
import dp_penalty

params = dp_penalty.PenaltyParams(
    tau = 4,
    prop_sigma = np.repeat(0.3, dim),
    r_clip_bound = 0.002,
    ocu = True,
    grw = True
)
