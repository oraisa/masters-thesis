
import numpy as np
import dp_penalty
params = dp_penalty.PenaltyParams(
    tau = 0.2,
    prop_sigma = np.repeat(0.00084, 30),#np.hstack((np.array((20, 7)), np.repeat(5, 28))) * 0.00024,
    r_clip_bound = 3,
    ocu = True,
    grw = True
)
