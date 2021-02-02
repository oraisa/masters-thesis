#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import mcmc_animation
import circle

parser = argparse.ArgumentParser()
parser.add_argument("results", type=str)
args = parser.parse_args()

df = pd.read_csv(
    args.results, names=[
        "Epsilon", "Delta", "experiment", "dim", "tempering", "i", "Algorithm", "Acceptance",
        "Clipping", "Grad Clipping",
        "MMD", "Mean Error", "Covariance Error"
    ]
)

algorithm_names = {"hmc": "HMC", "dpps": "DP Penalty"}
hue_order_short = ["hmc", "dpps"]
hue_order = [algorithm_names[algo] for algo in hue_order_short]
df["Algorithm"] = df["Algorithm"].map(algorithm_names)

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle("Circle")

sns.boxplot(x="Epsilon", y="Mean Error", hue="Algorithm", hue_order=hue_order, data=df, ax=axes[0])
axes[0].set_ylim(0)
sns.boxplot(x="Epsilon", y="Clipping", hue="Algorithm", hue_order=hue_order, data=df, ax=axes[1])
axes[1].set_ylim(0)
plt.savefig("../Thesis/figures/circle.pdf")
# plt.show()
