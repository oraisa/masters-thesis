import numpy as np
import dp_barker
params = dp_barker.BarkerParams(
        prop_sigma = np.array((0.008, 0.007)) * 0.1,
        batch_size = 1300
)