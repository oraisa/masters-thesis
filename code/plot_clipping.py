import jax.random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import banana_model
import experiments
import mmd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("results", type=str)
args = parser.parse_args()

df = pd.read_csv(
    args.results, names=[
        "Clip Bound", "dim", "i", "Algorithm", "Acceptance", "Clipping", "Disagreements", 
        "MMD", "Mean Error", "Variance Error"
    ]
)
df = df[df["Clip Bound"].isin([0.5, 1, 2, 4, 6, 8, 10, 1000])]

key = jax.random.PRNGKey(46237)
keys = jax.random.split(key, 10)

df2 = df[df["dim"] == 2]
problem2 = experiments.experiments["easy-2d"].get_problem()
mmds2 = [
    mmd.mmd(
        problem2.gen_true_posterior(1000, keys[i]),
        problem2.true_posterior
    ) for i in range(0, len(keys))
]
mmds2 = np.array(mmds2)

df10 = df[df["dim"] == 10]
problem10 = experiments.experiments["easy-10d"].get_problem()
mmds10= [
    mmd.mmd(
        problem10.gen_true_posterior(1000, keys[i]),
        problem10.true_posterior
    ) for i in range(1, len(keys))
]
mmds10 = np.array(mmds10)

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
sns.stripplot(x="Clip Bound", y="MMD", hue="Algorithm", hue_order=["HMC", "RWMH"], data=df2, ax=axes[0, 0])
for mmd in mmds2:
    axes[0, 0].axhline(mmd, linestyle="dashed", color="black")

sns.scatterplot(x="Clipping", y="MMD", hue="Algorithm", hue_order=["HMC", "RWMH"], data=df2, ax=axes[1, 0])
for mmd in mmds2:
    axes[1, 0].axhline(mmd, linestyle="dashed", color="black")

sns.stripplot(x="Clip Bound", y="MMD", hue="Algorithm", hue_order=["HMC", "RWMH"], data=df10, ax=axes[0, 1])
for mmd in mmds10:
    axes[0, 1].axhline(mmd, linestyle="dashed", color="black")

sns.scatterplot(x="Clipping", y="MMD", hue="Algorithm", hue_order=["HMC", "RWMH"], data=df10, ax=axes[1, 1])
for mmd in mmds10:
    axes[1, 1].axhline(mmd, linestyle="dashed", color="black")

axes[0, 0].set_title("d = 2")
axes[0, 1].set_title("d = 10")

axes[0, 1].set_ylim(0, 0.6)
axes[1, 0].set_ylim(0, 0.6)
axes[1, 1].set_ylim(0, 0.6)
axes[0, 0].set_ylim(0, 0.6)
plt.tight_layout()
plt.savefig("../Thesis/figures/clipping.pdf")
