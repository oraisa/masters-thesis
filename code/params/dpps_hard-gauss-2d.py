import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.07,
    prop_sigma = np.repeat(0.00015, 2),
    r_clip_bound = 50,
    ocu = False,
    grw = False
)
