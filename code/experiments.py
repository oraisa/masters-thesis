import numpy as np
import pandas as pd
import circle
import banana_model
import gauss_model
import jax.random

class BananaExperiment:
    def __init__(self, dim, n0, a, n, start_stdev):
        self.dim = dim
        self.n0 = n0
        self.a = a
        self.n = n
        self.start_stdev = start_stdev

        self.sigma2_0 = 1000

    def get_problem(self):
        return banana_model.get_problem(self.dim, self.n0, self.a, self.n)

class GaussExperiment:
    def __init__(self, dim, n, start_stdev):
        self.dim = dim
        self.n = n

        self.a = 0
        self.n0 = None
        self.start_stdev = start_stdev
        self.sigma2_0 = 100

    def get_problem(self):
        return gauss_model.get_problem(self.dim, self.n)

class CircleExperiment:
    def __init__(self):
        self.start_stdev = 0.3

        self.n = 100000
        self.dim = 2
        self.n0 = None
        self.a = 0.00001
        self.sigma2_0 = np.nan

    def get_problem(self):
        return circle.problem()

experiments = {
    "easy-2d": BananaExperiment(dim=2, n0=None, a=20, n=100000, start_stdev=0.02),
    "easy-10d": BananaExperiment(dim=10, n0=None, a=20, n=200000, start_stdev=0.02),
    "tempered-2d": BananaExperiment(dim=2, n0=1000, a=20, n=100000, start_stdev=0.15),
    "tempered-10d": BananaExperiment(dim=10, n0=1000, a=20, n=200000, start_stdev=0.15),
    "gauss-30d": BananaExperiment(dim=30, n0=None, a=0, n=200000, start_stdev=0.02),
    "hard-2d": BananaExperiment(dim=2, n0=None, a=350, n=150000, start_stdev=0.015),
    # "hard-gauss-6d": GaussExperiment(dim=6, n=200000, start_stdev=0.005),
    "hard-gauss-2d": GaussExperiment(dim=2, n=200000, start_stdev=0.003),
    "circle": CircleExperiment()
}
exp_names = {
    "easy-2d": "Flat banana, d = 2",
    "hard-2d": "Narrow banana",
    "easy-10d": "Flat banana, d = 10",
    "tempered-2d": "Tempered banana, d = 2",
    "tempered-10d": "Tempered banana, d = 10",
    "gauss-30d": "High dimensional Gauss",
    "hard-gauss-2d": "Correlated Gauss",
    "circle": "Circle",
}

if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "Name": [exp_names[key] for key in experiments.keys()],
            "Dim": [exp.dim for exp in experiments.values()],
            "n": [exp.n for exp in experiments.values()],
            "$n_0$": [exp.n0 for exp in experiments.values()],
            "Start Deviation": [exp.start_stdev for exp in experiments.values()],
            "a": [exp.a for exp in experiments.values()],
            r"$\sigma^2_0$": [exp.sigma2_0 for exp in experiments.values()],
        }
    )
    with open("../Thesis/model_params_table.tex", "w") as f:
        df.to_latex(
            f, index=False, label="model_params_table", escape=False, na_rep="",
            float_format="%.4g",
            caption=r"""
            Model parameters. $n_0$ determines tempering by \(T=\frac{n_0}{n}\).
            For missing $n_0$, \(T = 1\). Start deviation is the standard deviation
            of the random starting point in the DP experiments.
            """
        )
