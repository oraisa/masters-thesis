
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 0.6,
    prop_sigma = np.repeat(0.0001, 10),
    r_clip_bound = 3,
    batch_size = 1000,
    ocu = False,
    grw = False
)
