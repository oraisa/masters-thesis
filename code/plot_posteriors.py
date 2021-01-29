import numpy as np
import matplotlib.pyplot as plt
from experiments import experiments

exps_to_plot = ["easy-2d", "hard-2d", "tempered-2d", "circle", "hard-gauss-2d"]
plot_titles = [
    "Easy Banana, a = 20", "Hard Banana, a = 350", "Tempered Banana, a = 20",
    "Circle", "Correlated Gaussian"
]
fig, axes = plt.subplots(3, 2, figsize=(10, 13))

for i, exp in enumerate(exps_to_plot):
    problem = experiments[exp].get_problem()
    ind = np.unravel_index(i, axes.shape)
    problem.plot_density(axes[ind])
    axes[ind].set_title(plot_titles[i])

plt.savefig("../Thesis/figures/posterior_plots.pdf")
