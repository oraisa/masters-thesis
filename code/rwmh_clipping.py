import numpy as np
import scipy.stats as stats
import pandas as pd
import banana_model
import mmd
import argparse

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

banana = banana_model.BananaModel(dim=args.dim)
data = banana.generate_test_data()
n, data_dim = data.shape

iters = 2000
prop_sigma = 0.01
clip_bound = args.clip_bound
dim = args.dim
theta0 = np.zeros(dim)
theta0[1] = 2.95
chain = np.zeros((iters + 1, dim))
chain[0] = theta0

clipped = np.zeros(iters + 1)
clip_diff = np.zeros(iters + 1)
orig_ratios = np.zeros(iters + 1)
accepts = 0
diff_accepts = np.zeros(iters + 1)
llc = banana.log_likelihood_no_sum(data, theta0)
for i in range(iters):
    current = chain[i, :]
    prop = stats.norm.rvs(size=dim, loc=current, scale=prop_sigma)

    lpc = banana.log_prior(current)
    lpp = banana.log_prior(prop)
    llp = banana.log_likelihood_no_sum(data, prop)
    ratio = llp - llc
    orig_ratio = ratio
    orig_ratios[i + 1] = np.sum(orig_ratio)

    clip = clip_bound * np.sqrt(np.sum(current - prop)**2)
    clipped[i + 1] = np.sum(np.abs(ratio) > clip)
    ratio = np.clip(ratio, -clip, clip)
    clip_diff[i + 1] = np.sum(orig_ratio - ratio)

    lambd = np.sum(ratio) + lpp - lpc
    u = np.log(np.random.rand())
    accept = u < lambd
    orig_accept = u < np.sum(orig_ratio) + lpp - lpc
    if accept != orig_accept:
        diff_accepts[i + 1] = 1
    if accept:
        chain[i + 1, :] = prop
        accepts += 1
        llc = llp
    else:
        chain[i + 1, :] = current
    if (i + 1) % 100 == 0:
        print("Iteration: {}".format(i + 1))

acceptance = accepts / iters
clipping = np.sum(clipped / iters / n)
diff_decisions = np.sum(diff_accepts)

final_chain = chain[int((iters - 1) / 2) + 1:, :]
posterior = banana.generate_posterior_samples(2000, data, 1)
mmd = mmd.mmd(final_chain, posterior)
mean_error = np.sqrt(np.sum((np.mean(posterior, axis=0) - np.mean(final_chain, axis=0))**2))
var_error = np.sqrt(np.sum((np.var(posterior, axis=0) - np.var(final_chain, axis=0))**2))

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
