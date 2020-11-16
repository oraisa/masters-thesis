import numpy as np
from ite.cost.x_factory import co_factory
from ite.cost.x_kernel import Kernel

def mmd(samples, true_samples):
    kernel = Kernel({"name": "RBF", "sigma": 1})
    co = co_factory("BDMMD_UStat", mult=True, kernel=kernel)
    
    return co.estimation(samples, true_samples)

def mean_error(samples, true_samples):
    return np.sqrt(np.sum((np.mean(samples, axis=0) - np.mean(true_samples, axis=0))**2))

def cov_error(samples, true_samples):
    cov_errors = np.cov(samples, rowvar=False) - np.cov(true_samples, rowvar=False)
    return np.sqrt(np.sum(cov_errors**2))

