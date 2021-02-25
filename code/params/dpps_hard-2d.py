import numpy as np
import dp_penalty

params = dp_penalty.PenaltyParams(
    tau = 0.20,
    prop_sigma = np.repeat(0.0015, 2),
    r_clip_bound = 5.5,
    ocu = False,
    grw = False
)
