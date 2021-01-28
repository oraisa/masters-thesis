import numpy as np
import dp_barker
params = dp_barker.BarkerParams(
    prop_sigma = np.repeat(0.008 * 15, 2),
    batch_size = 1300
)
