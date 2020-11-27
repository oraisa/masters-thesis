import numpy as np
import numba
from ite.cost.x_factory import co_factory
from ite.cost.x_kernel import Kernel
import timeit

def mmd(samples, true_samples):
    subset1 = samples[np.random.choice(samples.shape[0], 50, replace=True), :]
    subset2 = true_samples[np.random.choice(true_samples.shape[0], 50, replace=True), :]
    distances = np.sqrt(np.sum((subset1 - subset2)**2))

    kernel = Kernel({"name": "RBF", "sigma": np.median(distances)})
    co = co_factory("BDMMD_UStat", mult=True, kernel=kernel)
    
    return co.estimation(samples, true_samples)

def mean_error(samples, true_samples):
    return np.sqrt(np.sum((np.mean(samples, axis=0) - np.mean(true_samples, axis=0))**2))

def cov_error(samples, true_samples):
    cov_errors = np.cov(samples, rowvar=False) - np.cov(true_samples, rowvar=False)
    return np.sqrt(np.sum(cov_errors**2))

sigma = 2
@numba.njit
def kernel(x1, x2):
    return np.exp(-np.sum((x1 - x2)**2) / (2 * sigma**2))

@numba.njit
def alt_mmd(sample1, sample2):
    n = sample1.shape[0]
    m = sample2.shape[0]

    term1 = 0
    for i in range(0, n):
        for j in range(i + 1, n):
            term1 += kernel(sample1[i, :], sample1[j, :])
    term2 = 0
    for i in range(0, m):
        for j in range(i + 1, m):
            term2 += kernel(sample2[i, :], sample2[j, :])
    term3 = 0
    for i in range(n):
        for j in range(m):
            term3 += kernel(sample1[i, :], sample2[j, :])
    return 2 * term1 / (n * (n - 1)) + 2 * term2 / (m * (m - 1)) - 2 * term3 / (n * m)

if __name__ == "__main__":
    n = 1000
    m = 1000
    sample1 = np.random.randn(n).reshape((m, 1))# + np.ones((100, 1))
    sample2 = np.random.randn(m).reshape((m, 1))
    print("MMD: {}".format(mmd(sample1, sample2)))
    mmd_res = alt_mmd(sample1, sample2)
    print("Alt MMD: {}".format(np.sqrt(np.abs(mmd_res))))
    res1 = timeit.timeit(lambda: mmd(sample1, sample2), number=1)
    # res2 = timeit.timeit(lambda: alt_mmd(sample1, sample2), number=1)
    res3 = timeit.timeit(lambda: alt_mmd(sample1, sample2), number=1)
    print(res1)
    # print(res2)
    print(res3)
