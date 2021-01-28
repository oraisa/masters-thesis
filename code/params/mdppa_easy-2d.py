
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 0.9,
    prop_sigma = np.repeat(0.008 * 0.1, 2),
    r_clip_bound = 1.5,
    batch_size = 1000,
    ocu = True,
    grw = True
)
