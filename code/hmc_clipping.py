import numpy as np
import scipy.stats as stats
import pandas as pd
import banana_model
import mmd
import argparse
import clipping

parser = argparse.ArgumentParser()
parser.add_argument("clip_bound", type=float)
parser.add_argument("dim", type=int)
parser.add_argument("index", type=int)
parser.add_argument("output", type=str)
args = parser.parse_args()

problem = banana_model.get_problem(dim=args.dim, a=20, n0=None, n=100000)
n, data_dim = problem.data.shape
dim = args.dim

iters = 500
eta = 0.001
L = 15
mass = np.ones(dim)
mass[1] = 0.5
if dim > 2:
    mass[2:] = 0.4
clip_bound = args.clip_bound
theta0 = np.zeros(dim)
theta0[1] = 3

# Set the seed for the starting points only based on index
np.random.seed(4238946 + args.index)
problem.theta0 += np.random.normal(scale=0.05, size=dim)

# Set the seed for the algorithm to be different for each algorithm
np.random.seed(
    (int(("hmc" + str(args.clip_bound)).encode("utf8").hex(), 16) + args.index) % 2**32
)

res = clipping.hmc(problem, iters, eta, L, mass, clip_bound, theta0)

acceptance = res.accepts / iters
clipping = np.sum(res.clipped / iters / n)
diff_decisions = np.sum(res.diff_accepts)

posterior = problem.true_posterior
mmd = mmd.mmd(res.final_chain, posterior)
mean_error = np.sqrt(np.sum((np.mean(posterior, axis=0) - np.mean(res.final_chain, axis=0))**2))
var_error = np.sqrt(np.sum((np.var(posterior, axis=0) - np.var(res.final_chain, axis=0))**2))

result = pd.DataFrame({
    "clip bound": [clip_bound],
    "dim": [dim],
    "i": [args.index],
    "algo": ["HMC"],
    "acceptance": [acceptance],
    "clipping": [clipping],
    "diff decisions": [diff_decisions],
    "mmd": [mmd],
    "mean error": [mean_error],
    "var error": [var_error]
})
result.to_csv(args.output, header=False)
