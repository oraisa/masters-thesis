
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 1.6,
    prop_sigma = np.repeat(8 * 0.0035, 10),
    r_clip_bound = 4,
    batch_size = 1000,
    ocu = True,
    grw = True
)
