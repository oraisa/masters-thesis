
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 0.7,
    prop_sigma = np.repeat(8 * 0.0025, 10),
    r_clip_bound = 2,
    batch_size = 1200,
    ocu = False,
    grw = False
)
