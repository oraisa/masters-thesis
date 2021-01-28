
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 2.0,
    prop_sigma = np.repeat(0.04, 2),
    r_clip_bound = 4,
    batch_size = 1000,
    ocu = True,
    grw = True
)
