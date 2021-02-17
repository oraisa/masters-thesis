import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("banana_results", type=str)
parser.add_argument("circle_results", type=str)
args = parser.parse_args()

names= [
    "Epsilon", "Delta", "experiment", "dim", "tempering", "i", "Algorithm", "Acceptance",
    "Clipping", "Grad Clipping",
    "MMD", "Mean Error", "Covariance Error"
]
df = pd.read_csv(args.banana_results, names=names)

algorithm_names = {
    "hmc": "DP HMC", "dpps": "DP Penalty", "dppa": "DP Penalty OCU+GWMH",
    "mdpps": "Minibatch DP Penalty", "mdppa": "Minibatch DP Penalty OCU+GWMH",
    "barker": "DP Barker"
}
df["Algorithm"] = df["Algorithm"].map(algorithm_names)

df_easy_2d = df[df["experiment"] == "easy-2d"]
df_easy_10d = df[df["experiment"] == "easy-10d"]
df_tempered_2d = df[df["experiment"] == "tempered-2d"]
df_tempered_10d = df[df["experiment"] == "tempered-10d"]
df_gauss_30d = df[df["experiment"] == "gauss-30d"]
df_gauss_hard_6d = df[df["experiment"] == "hard-gauss-6d"]
df_gauss_hard_2d = df[df["experiment"] == "hard-gauss-2d"]
df_hard_2d = df[df["experiment"] == "hard-2d"]

df_circle = pd.read_csv(args.circle_results, names=names)
df_circle["Algorithm"] = df_circle["Algorithm"].map(algorithm_names)

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


plot_data(
    "Grad Clipping", ["DP HMC"],
    [
        [df_easy_2d, df_tempered_2d], [df_easy_10d, df_tempered_10d],
        [df_gauss_30d, df_gauss_hard_2d], [df_hard_2d, df_circle]
    ],
    np.array((
        ("d = 2, non-tempered", "d = 2, tempered"), ("d = 10, non-tempered", "d = 10, tempered"),
        ("d = 30, Gaussian", "d = 2, correlated Gaussian"),
        ("d = 2, narrow banana", "Circle")
    )),
    (0, 0.5), "../Thesis/figures/grad_clipping.pdf",
    legend_outside=False
)
