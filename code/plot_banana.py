#!/usr/bin/env python3

import jax.random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import banana_model
import mmd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("results", type=str)
args = parser.parse_args()

df = pd.read_csv(
    args.results, names=[
        "Epsilon", "Delta", "dim", "i", "Algorithm", "Acceptance", "Clipping", "Grad Clipping",
        "MMD", "Mean Error", "Covariance Error"
    ]
)

key = jax.random.PRNGKey(46237)
keys = jax.random.split(key, 10)

df2 = df[df["dim"] == 2]
banana2 = banana_model.BananaModel(dim=2)
data2 = banana2.generate_test_data()
posterior2 = banana2.generate_posterior_samples(2000, data2, 1, keys[0])
mmds2 = [mmd.mmd(banana2.generate_posterior_samples(1000, data2, 1, keys[i]), posterior2) for i in range(1, len(keys))]
mmds2 = np.array(mmds2)

# df10 = df[df["dim"] == 10]
# df10 = df10[df10["Algorithm"] == "HMC"]
# banana10 = banana_model.BananaModel(dim=10)
# data10 = banana10.generate_test_data()
# posterior10 = banana10.generate_posterior_samples(2000, data10, 1, keys[0])
# mmds10= [mmd.mmd(banana10.generate_posterior_samples(1000, data10, 1, keys[i]), posterior10) for i in range(1, len(keys))]
# mmds10 = np.array(mmds10)

fig, axes = plt.subplots(1, 1, figsize=(10, 5))
sns.stripplot(x="Epsilon", y="MMD", hue="Algorithm", data=df2, ax=axes)
for mmd in mmds2:
    axes.axhline(mmd, linestyle="dashed", color="black")

# sns.scatterplot(x="Clipping", y="MMD", hue="Algorithm", data=df2, ax=axes[1, 0])
# for mmd in mmds2:
#     axes[1, 0].axhline(mmd, linestyle="dashed", color="black")

# sns.stripplot(x="Clip Bound", y="MMD", hue="Algorithm", data=df10, ax=axes[0, 1])
# for mmd in mmds10:
#     axes[0, 1].axhline(mmd, linestyle="dashed", color="black")

# sns.scatterplot(x="Clipping", y="MMD", hue="Algorithm", data=df10, ax=axes[1, 1])
# for mmd in mmds10:
#     axes[1, 1].axhline(mmd, linestyle="dashed", color="black")

# axes[0, 0].set_title("d = 2")
# axes[0, 1].set_title("d = 10")
plt.tight_layout()
plt.savefig("../Thesis/figures/banana.pdf")
plt.show()
