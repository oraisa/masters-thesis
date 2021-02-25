
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 0.5,
    prop_sigma = np.repeat(8 * 0.00004, 10),
    r_clip_bound = 2.0,
    batch_size = 1000,
    ocu = True,
    grw = True
)
