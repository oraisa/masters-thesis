import jax
import jax.numpy as np 
import jax.random as random
import matplotlib.pyplot as plt
import util
jax.config.update("jax_enable_x64", True)

def log_likelihood_per_sample(theta, r):
    x = theta[0]
    y = theta[1]
    return -0.00001 * (x**2 + y**2 - r[0]**2)**2

def problem():
    return util.Problem(
        log_likelihood_per_sample, lambda x: 0.0, generate_data(),
        1, np.array((0.0, 1.0)), None, plot_density, np.zeros(2)
    )

def generate_data():
    key = random.PRNGKey(742983)
    key, subkey = random.split(key)
    R = random.normal(subkey, shape=(100000, 1)) + 3
    return R

def plot_density(problem, ax):
    def circle_dens(x, y, R):
        return np.exp(problem.log_likelihood(np.array((x, y)), R))

    R = problem.data
    xs = np.linspace(-5, 5, 100)
    ys = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(xs, ys)
    z = jax.vmap(jax.vmap(circle_dens, in_axes=(0, 0, None)), in_axes=(0, 0, None))(x, y, R)
    ax.contour(x, y, z, 20)

if __name__ == "__main__":
    problem = problem()

    fig, ax = plt.subplots()
    problem.plot_density(ax)
    plt.show()


