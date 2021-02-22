import itertools
import jax
import argparse
import jax.numpy as np
import pandas as pd
import banana_model
import gauss_model
import experiments
import mmd

parser = argparse.ArgumentParser()
parser.add_argument("output", type=str)
args = parser.parse_args()

key = jax.random.PRNGKey(4623878)
keys = jax.random.split(key, 10)

def mmd_for_experiment(problem, i):
    posterior = problem.gen_true_posterior(1000, keys[i])
    return mmd.mmd(posterior, problem.true_posterior)

problems = list(filter(
    lambda par: par[1].gen_true_posterior is not None,
    map(lambda par: (par[0], par[1].get_problem()), experiments.experiments.items())
))
mmds = list(itertools.chain.from_iterable(map(
    lambda par: (par[0], mmd_for_experiment(par[1], i)),
    problems
) for i in range(10)))

df = pd.DataFrame(mmds, columns=["experiment", "MMD"])
df.to_csv(args.output)
