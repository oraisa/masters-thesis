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
    "hmc": "DP HMC", "dpps": "DP Penalty", "dppa": "DP Penalty OCU+GWMH",
    "mdpps": "Minibatch DP Penalty", "mdppa": "Minibatch DP Penalty OCU+GWMH",
    "barker": "DP Barker"
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
df_gauss_hard_2d = df[df["experiment"] == "hard-gauss-2d"]
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
#
def plot_data(y, hue_order, datasets, titles, ylim, filename, legend_outside=True):
    h, w = titles.shape
    fig, axes = plt.subplots(h, w, figsize=(10, 13), squeeze=False)
    for j in range(w):
        for i in range(h):
            if datasets[i][j] is not None:
                axes[i, j].set_title(titles[i, j])
                sns.boxplot(
                    x="Epsilon", y=y, hue="Algorithm", hue_order=hue_order,
                    data=datasets[i][j], ax=axes[i, j]
                )
                axes[i, j].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
                axes[i, j].set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()

def plot_mmd_clip_acceptance(hue_order, datasets, titles, mmd_file, clip_file, acc_file):
    plot_data("MMD", hue_order, datasets, titles, (0, 0.9), mmd_file)
    plot_data("Clipping", hue_order, datasets, titles, 0, clip_file)
    plot_data("Acceptance", hue_order, datasets, titles, 0, acc_file)

plot_mmd_clip_acceptance(
    hue_order, [[df_easy_2d], [df_tempered_2d], [df_easy_10d], [df_tempered_10d]],
    np.array(
        ("d = 2, non-tempered", "d = 2, tempered",
         "d = 10, non-tempered", "d = 10, tempered")
    ).reshape((4, 1)),
    "../Thesis/figures/banana_mmd.pdf",
    "../Thesis/figures/banana_clipping.pdf",
    "../Thesis/figures/banana_acceptance.pdf"
)

plot_mmd_clip_acceptance(
    extra_hue_order, [[df_gauss_30d], [df_hard_2d], [df_gauss_hard_2d]],
    np.array(("d = 30, Gaussian", "d = 2, narrow banana", "d = 2, correlated Gaussian")).reshape((3, 1)),
    "../Thesis/figures/banana_extra.pdf",
    "../Thesis/figures/banana_extra_clipping.pdf",
    "../Thesis/figures/banana_extra_acceptance.pdf"
)

plot_data(
    "Grad Clipping", ["DP HMC"],
    [
        [df_easy_2d, df_tempered_2d], [df_easy_10d, df_tempered_10d],
        [df_gauss_30d, df_gauss_hard_2d], [df_hard_2d, None]
    ],
    np.array((
        ("d = 2, non-tempered", "d = 2, tempered"), ("d = 10, non-tempered", "d = 10, tempered"),
        ("d = 30, Gaussian", "d = 2, correlated Gaussian"),
        ("d = 2, narrow banana", None)
    )),
    (0, 0.5), "../Thesis/figures/banana_grad_clipping.pdf",
    legend_outside=False
)
