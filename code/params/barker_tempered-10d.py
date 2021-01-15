import numpy as np
import dp_barker
params = dp_barker.BarkerParams(
        prop_sigma = np.array((8, 7, 5, 5, 5, 5, 5, 5, 5, 5)) * 0.005,
        batch_size = 2600
)
