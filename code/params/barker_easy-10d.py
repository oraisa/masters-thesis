import numpy as np
import dp_barker
params = dp_barker.BarkerParams(
    prop_sigma = np.repeat(8 * 0.0001, 10),
    batch_size = 2600
)
