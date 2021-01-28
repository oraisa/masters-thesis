
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 1.0,
    prop_sigma = np.repeat(8 * 0.002, 10),
    r_clip_bound = 3,
    batch_size = 1000,
    ocu = False,
    grw = False
)
