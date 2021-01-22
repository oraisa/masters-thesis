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
        "Epsilon", "Delta", "experiment", "dim", "tempering", "i", "Algorithm", "Acceptance",
        "Clipping", "Grad Clipping",
        "MMD", "Mean Error", "Covariance Error"
    ]
)

algorithm_names = {
    "hmc": "HMC", "dpps": "DP Penalty", "dppa": "DP Penalty Advanced",
    "mdpps": "Minibatch DP Penalty", "mdppa": "Minibatch DP Penalty Advanced",
    "barker": "Barker"
}
hue_order_short = ["hmc", "dpps", "dppa", "mdpps", "mdppa", "barker"]
hue_order = [algorithm_names[algo] for algo in hue_order_short]
extra_hue_order = hue_order[0:3]
df["Algorithm"] = df["Algorithm"].map(algorithm_names)

key = jax.random.PRNGKey(46237)
keys = jax.random.split(key, 10)

df_easy_2d = df[df["experiment"] == "easy-2d"]
df_easy_10d = df[df["experiment"] == "easy-10d"]
df_tempered_2d = df[df["experiment"] == "tempered-2d"]
df_tempered_10d = df[df["experiment"] == "tempered-10d"]
df_gauss_30d = df[df["experiment"] == "gauss-30d"]
df_gauss_hard_6d = df[df["experiment"] == "hard-gauss-6d"]
df_hard_2d = df[df["experiment"] == "hard-2d"]

# banana2 = banana_model.BananaModel(dim=2# )
# data2 = banana2.generate_test_data()
# posterior2 = banana2.generate_posterior_samples(2000, data2, 1, keys[0])
# mmds2 = [mmd.mmd(banana2.generate_posterior_samples(1000, data2, 1, keys[i]), posterior2) for i in range(1, len(keys))]
# mmds2 = np.array(mmds2)

# df10 = df10[df10["Algorithm"] == "HMC"]
# banana10 = banana_model.BananaModel(dim=10)
# data10 = banana10.generate_test_data()
# posterior10 = banana10.generate_posterior_samples(2000, data10, 1, keys[0])
# mmds10= [mmd.mmd(banana10.generate_posterior_samples(1000, data10, 1, keys[i]), posterior10) for i in range(1, len(keys))]
# mmds10 = np.array(mmds10)

fig, axes = plt.subplots(4, 1, figsize=(10, 13))
axes[0].set_title("d = 2, non-tempered")
sns.boxplot(x="Epsilon", y="MMD", hue="Algorithm", hue_order=hue_order, data=df_easy_2d, ax=axes[0])
axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[0].set_ylim(0, 0.9)

axes[1].set_title("d = 2, tempered")
sns.boxplot(x="Epsilon", y="MMD", hue="Algorithm", hue_order=hue_order, data=df_tempered_2d, ax=axes[1])
axes[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[1].set_ylim(0, 0.9)

axes[2].set_title("d = 10, non-tempered")
sns.boxplot(x="Epsilon", y="MMD", hue="Algorithm", hue_order=hue_order, data=df_easy_10d, ax=axes[2])
axes[2].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[2].set_ylim(0, 0.9)

axes[3].set_title("d = 10, tempered")
sns.boxplot(x="Epsilon", y="MMD", hue="Algorithm", hue_order=hue_order, data=df_tempered_10d, ax=axes[3])
axes[3].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[3].set_ylim(0, 0.9)

plt.tight_layout()
plt.savefig("../Thesis/figures/banana_mmd.pdf")
# plt.show()

fig, axes = plt.subplots(4, 1, figsize=(10, 13))
axes[0].set_title("d = 2, non-tempered")
sns.boxplot(x="Epsilon", y="Clipping", hue="Algorithm", hue_order=hue_order, data=df_easy_2d, ax=axes[0])
axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[0].set_ylim(0)

axes[1].set_title("d = 2, tempered")
sns.boxplot(x="Epsilon", y="Clipping", hue="Algorithm", hue_order=hue_order, data=df_tempered_2d, ax=axes[1])
axes[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[1].set_ylim(0)

axes[2].set_title("d = 10, non-tempered")
sns.boxplot(x="Epsilon", y="Clipping", hue="Algorithm", hue_order=hue_order, data=df_easy_10d, ax=axes[2])
axes[2].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[2].set_ylim(0)

axes[3].set_title("d = 10, tempered")
sns.boxplot(x="Epsilon", y="Clipping", hue="Algorithm", hue_order=hue_order, data=df_tempered_10d, ax=axes[3])
axes[3].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[3].set_ylim(0)

plt.tight_layout()
plt.savefig("../Thesis/figures/banana_clipping.pdf")
# plt.show()

fig, axes = plt.subplots(3, 1, figsize=(10, 13))
axes[0].set_title("d = 30, Gaussian")
sns.boxplot(x="Epsilon", y="MMD", hue="Algorithm", hue_order=extra_hue_order, data=df_gauss_30d, ax=axes[0])
axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[0].set_ylim(0, 0.9)

axes[1].set_title("d = 2, hard banana")
sns.boxplot(x="Epsilon", y="MMD", hue="Algorithm", hue_order=extra_hue_order, data=df_hard_2d, ax=axes[1])
axes[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[1].set_ylim(0, 0.9)

axes[2].set_title("d = 6, hard Gaussian")
sns.boxplot(x="Epsilon", y="MMD", hue="Algorithm", hue_order=extra_hue_order, data=df_gauss_hard_6d, ax=axes[2])
axes[2].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[2].set_ylim(0, 0.9)

plt.tight_layout()
plt.savefig("../Thesis/figures/banana_extra.pdf")
# plt.show()

fig, axes = plt.subplots(3, 1, figsize=(10, 13))
axes[0].set_title("d = 30, Gaussian")
sns.boxplot(x="Epsilon", y="Clipping", hue="Algorithm", hue_order=extra_hue_order, data=df_gauss_30d, ax=axes[0])
axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[0].set_ylim(0)

axes[1].set_title("d = 2, hard banana")
sns.boxplot(x="Epsilon", y="Clipping", hue="Algorithm", hue_order=extra_hue_order, data=df_hard_2d, ax=axes[1])
axes[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[1].set_ylim(0)

axes[2].set_title("d = 6, hard Gaussian")
sns.boxplot(
    x="Epsilon", y="Clipping", hue="Algorithm", hue_order=extra_hue_order,
    data=df_gauss_hard_6d, ax=axes[2]
)
axes[2].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
axes[2].set_ylim(0)

plt.tight_layout()
plt.savefig("../Thesis/figures/banana_extra_clipping.pdf")
# plt.show()

fig, axes = plt.subplots(4, 2, figsize=(10, 13))
axes[0, 0].set_title("d = 2, non-tempered")
sns.boxplot(x="Epsilon", y="Grad Clipping", hue="Algorithm", hue_order=["HMC"], data=df_easy_2d, ax=axes[0, 0])
axes[0, 0].set_ylim(0, 0.5)

axes[0, 1].set_title("d = 2, tempered")
sns.boxplot(x="Epsilon", y="Grad Clipping", hue="Algorithm", hue_order=["HMC"], data=df_tempered_2d, ax=axes[0, 1])
axes[0, 1].set_ylim(0, 0.5)

axes[1, 0].set_title("d = 10, non-tempered")
sns.boxplot(x="Epsilon", y="Grad Clipping", hue="Algorithm", hue_order=["HMC"], data=df_easy_10d, ax=axes[1, 0])
axes[1, 0].set_ylim(0, 0.5)

axes[1, 1].set_title("d = 10, tempered")
sns.boxplot(x="Epsilon", y="Grad Clipping", hue="Algorithm", hue_order=["HMC"], data=df_tempered_10d, ax=axes[1, 1])
axes[1, 1].set_ylim(0, 0.5)

axes[2, 0].set_title("d = 30, Gaussian")
sns.boxplot(x="Epsilon", y="Grad Clipping", hue="Algorithm", hue_order=["HMC"], data=df_gauss_30d, ax=axes[2, 0])
axes[2, 0].set_ylim(0, 0.5)

axes[2, 1].set_title("d = 6, hard Gaussian")
sns.boxplot(x="Epsilon", y="Grad Clipping", hue="Algorithm", hue_order=["HMC"], data=df_gauss_hard_6d, ax=axes[2, 1])
axes[2, 1].set_ylim(0, 0.5)

axes[3, 0].set_title("d = 2, hard banana")
sns.boxplot(x="Epsilon", y="Grad Clipping", hue="Algorithm", hue_order=["HMC"], data=df_hard_2d, ax=axes[3, 0])
axes[3, 0].set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig("../Thesis/figures/banana_grad_clipping.pdf")
#plt.show()
