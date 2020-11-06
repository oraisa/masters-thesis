import jax
import jax.numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import jax.scipy.stats as stats 
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing

def load_data():
    data = pd.read_csv("abalone.csv")
    data["Sex"] = 1 * (data["Sex"] == "M")
    data["target"] = 1 * (data["Rings"] > 9)
    data.drop("Rings", inplace=True, axis=1)

    X = data.drop("target", axis=1).values
    y = data["target"].values
    y = y * 2 - 1

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42374)

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    lr = sklearn.linear_model.LogisticRegression(fit_intercept=False, C=1000)
    lr.fit(X_train, y_train)
    print("Test score: {}".format(lr.score(X_test, y_test)))

@jax.jit
def log_prior(theta):
    return np.sum(stats.norm.logpdf(theta, scale=100))

@jax.jit
def log_likelihood(theta, X, y):
    return np.sum(log_likelihood_no_sum(theta, X, y))

@jax.jit 
def log_likelihood_no_sum(theta, X, y):
    p = np.matmul(X, theta.reshape((-1, 1))).reshape((-1,))
    return -np.log(1 + np.exp(-y * p))

@jax.jit
def log_target(theta, X, y):
    return log_likelihood(theta, X, y) + log_prior(theta)

log_target_grad = jax.jit(jax.grad(log_target, 0))

