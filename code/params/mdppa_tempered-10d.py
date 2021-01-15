
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
        tau = 0.6,
        prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.0035,
        r_clip_bound = 1,
        batch_size = 1000,
        ocu = True,
        grw = True
)
