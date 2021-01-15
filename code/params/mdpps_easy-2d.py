
import numpy as np
import dp_penalty_minibatch
params = dp_penalty_minibatch.MinibatchPenaltyParams(
        tau = 0.9,
        prop_sigma = np.array((0.008, 0.007)) * 0.2,
        r_clip_bound = 1,
        batch_size = 1000,
        ocu = False,
        grw = False
)
