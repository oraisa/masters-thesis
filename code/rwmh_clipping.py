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


np.random.seed(46237)
for i in range(args.index + 1):
    seed = np.random.randint(2**32)
np.random.seed(seed)

problem = banana_model.get_problem(dim=args.dim, a=20, n0=None, n=100000)
n, data_dim = problem.data.shape

iters = 3000
prop_sigma = 0.01 if args.dim == 2 else np.hstack(
    (np.array((0.015, 0.015)), np.repeat(0.002, args.dim - 2))
)
clip_bound = args.clip_bound
dim = args.dim
theta0 = np.zeros(dim)
theta0[1] = 3
theta0 += np.random.normal(scale=0.05, size=dim)

res = clipping.rwmh(problem, iters, prop_sigma, clip_bound, theta0)

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
    "algo": ["RWMH"],
    "acceptance": [acceptance],
    "clipping": [clipping],
    "diff decisions": [diff_decisions],
    "mmd": [mmd],
    "mean error": [mean_error],
    "var error": [var_error]
})
result.to_csv(args.output, header=False)

