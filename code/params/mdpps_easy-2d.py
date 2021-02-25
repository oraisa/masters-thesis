
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
    tau = 0.7,
    prop_sigma = np.repeat(0.0010, 2),
    r_clip_bound = 1,
    batch_size = 1000,
    ocu = False,
    grw = False
)
