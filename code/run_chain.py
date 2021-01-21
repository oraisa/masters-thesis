import numpy as np
import pandas as pd
import argparse
import mmd
import banana_model
import dp_penalty
import dp_penalty_minibatch
import hmc
import dp_barker
import params

class Experiment:
    def __init__(self, dim, n0, a, n):
        self.dim = dim
        self.n0 = n0
        self.a = a
        self.n = n

parser = argparse.ArgumentParser()
parser.add_argument("algorithm", type=str)
parser.add_argument("params", type=str)
parser.add_argument("experiment", type=str)
parser.add_argument("epsilon", type=float)
parser.add_argument("index", type=int)
parser.add_argument("output", type=str)
args = parser.parse_args()

algorithms = {
    "dpps": dp_penalty.dp_penalty,
    "dppa": dp_penalty.dp_penalty,
    "mdpps": dp_penalty_minibatch.dp_penalty_minibatch,
    "mdppa": dp_penalty_minibatch.dp_penalty_minibatch,
    "barker": dp_barker.dp_barker,
    "hmc": hmc.hmc
}
experiments = {
    "easy-2d": Experiment(dim=2, n0=None, a=20, n=100000),
    "hard-2d": Experiment(dim=2, n0=None, a=40, n=100000),
    "easy-10d": Experiment(dim=10, n0=None, a=20, n=200000),
    "tempered-2d": Experiment(dim=2, n0=1000, a=5, n=100000),
    "tempered-10d": Experiment(dim=10, n0=1000, a=5, n=200000),
    "gauss-30d": Experiment(dim=30, n0=None, a=0, n=200000)
}
exp = experiments[args.experiment]
problem = banana_model.get_problem(exp.dim, exp.n0, exp.a, exp.n)

# Set the seed for the starting points only based on index
np.random.seed(53274257 + args.index)
problem.theta0 += np.random.normal(scale=0.02, size=problem.dim)

# Set the seed for the algorithm to be different for each algorithm
np.random.seed(
    (int((
        args.algorithm + args.experiment + str(args.epsilon)
    ).encode("utf8").hex(), 16) + args.index) % 2**32
)

posterior = problem.true_posterior
dim = problem.dim
n, data_dim = problem.data.shape
epsilon = args.epsilon
delta = 0.1 / n

par = params.__dict__[args.params].params
res = algorithms[args.algorithm](problem, epsilon, delta, par, verbose=False)

result = pd.DataFrame({
    "epsilon": [epsilon],
    "delta": [delta],
    "experiment": [args.experiment],
    "dim": [dim],
    "tempering": [problem.temp_scale != 1],
    "i": [args.index],
    "algo": [args.algorithm],
    "acceptance": [res.acceptance],
    "r_clipping": [res.clipped_r],
    "grad_clipping": [res.clipped_grad],
    "mmd": [res.mmd],
    "mean error": [res.mean_error],
    "cov error": [res.cov_error]
})
result.to_csv(args.output, header=False, index=False)
