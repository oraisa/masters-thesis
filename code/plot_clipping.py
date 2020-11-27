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
        "clip bound", "dim", "i", "algo", "acceptance", "clipping", "diff decisions", 
        "mmd", "mean error", "var error"
    ]
)

key = jax.random.PRNGKey(46237)
keys = jax.random.split(key, 10)

banana2 = banana_model.BananaModel(dim=2)
data2 = banana2.generate_test_data()
posterior2 = banana2.generate_posterior_samples(2000, data2, 1, keys[0])
mmds2 = [mmd.mmd(banana2.generate_posterior_samples(1000, data2, 1, keys[i]), posterior2) for i in range(1, len(keys))]
mmds2 = np.array(mmds2)

banana10 = banana_model.BananaModel(dim=10)
data10 = banana10.generate_test_data()
posterior10 = banana10.generate_posterior_samples(2000, data10, 1, keys[0])
mmds10= [mmd.mmd(banana10.generate_posterior_samples(1000, data10, 1, keys[i]), posterior10) for i in range(1, len(keys))]
mmds10 = np.array(mmds10)

sns.catplot(x="clip bound", y="mmd", hue="dim", kind="strip", data=df, height=5, aspect=2)
for mmd in mmds2:
    plt.axhline(mmd, linestyle="dashed", color="black")
for mmd in mmds10:
    plt.axhline(mmd, linestyle="dashed", color="green")
plt.show()

sns.relplot(x="clipping", y="mmd", hue="dim", kind="scatter", data=df, height=5, aspect=2)
for mmd in mmds2:
    plt.axhline(mmd, linestyle="dashed", color="black")
for mmd in mmds10:
    plt.axhline(mmd, linestyle="dashed", color="green")
plt.show()
