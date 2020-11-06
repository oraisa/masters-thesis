import pymc3 as pm
import numpy as np
import arviz
import matplotlib.pyplot as plt
import seaborn as sns
import abalone_lr as abalone
import pickle

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = abalone.load_data()
    y_train = (y_train + 1) / 2
    y_test = (y_test + 1) / 2
    n, dim = X_train.shape

    with pm.Model() as model:
        theta = pm.Normal("theta", mu=0, sd=100, shape=dim)
        alpha = pm.math.dot(X_train, theta)
        y = pm.Bernoulli("y", logit_p=alpha, observed=y_train)

        approx = pm.fit(n=100000, method="fullrank_advi")
        trace = approx.sample(1000)
        samples = trace.get_values("theta")

        map = pm.find_MAP()["theta"]

        L_1d = approx.params[0].get_value()
        mu = approx.params[1].get_value()
        L = np.zeros((dim, dim))
        L[np.tril_indices(dim)] = L_1d

        with open("abalone_var_params.p", "wb") as file:
            pickle.dump({"mu": mu, "L": L}, file)

        alternative_samples = (L @ np.random.normal(size=(dim, 1000))).transpose() + mu

        mcmc_trace = pm.sample(1000, tune=500, chains=4, cores=1)
        print(pm.summary(mcmc_trace))
        arviz.to_netcdf(mcmc_trace, "abalone_post.nc")
        arviz.plot_trace(mcmc_trace)
        plt.show()

        mcmc_trace = arviz.from_netcdf("abalone_post.nc")
        mcmc_samples = np.concatenate(mcmc_trace.posterior.theta.values, axis=0)
        fig, axes = plt.subplots(dim)
        for i in range(dim):
            sns.kdeplot(mcmc_samples[:, i], ax=axes[i])
            sns.kdeplot(samples[:, i], ax=axes[i])
            sns.kdeplot(alternative_samples[:, i], ax=axes[i])
            axes[i].axvline(map[i], linestyle="dashed", color="black")
        plt.show()
