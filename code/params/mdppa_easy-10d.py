
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 0.6,
    prop_sigma = np.repeat(8 * 0.00006, 10),
    r_clip_bound = 3.5,
    batch_size = 1000,
    ocu = True,
    grw = True
)
