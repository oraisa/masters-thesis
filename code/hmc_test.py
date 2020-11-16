import jax.numpy as np
import numpy.random as npr
import jax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import circle

problem = circle.problem()
def target(theta):
    return problem.log_likelihood(theta, problem.data) + problem.log_prior(theta)
 
def gradF(theta):
    # return np.sum(problem.log_likelihood_grads(theta, problem.data), axis=0)
    return problem.log_likelihood_grad_clipped(10, theta, problem.data)[0]

def hmc(theta0, M, target, epsilon, L):
    thetas = np.zeros([M, len(theta0)])
    leapfrog_chain = np.zeros((M * L, len(theta0)))
    # gradF = jax.grad(target)
    theta = theta0
    g = gradF(theta)  # set gradient using initial theta
    logP = target(theta)  # set objective function too
    accepts = 0
    for m in range(M): # draw M samples
        p = npr.normal(size=theta.shape)  # initial momentum is Normal(0,1)
        H = p.T @ p / 2 - logP   # evaluate H(theta,p)
        thetanew = theta
        gnew = g
        for l in range(L): # make L 'leapfrog' steps
            p = p + epsilon * gnew / 2   # make half-step in p
            thetanew = thetanew + epsilon * p    # make step in theta
            leapfrog_chain = jax.ops.index_update(leapfrog_chain, jax.ops.index[m*L + l,:], thetanew)
            gnew = gradF(thetanew)           # find new gradient
            p = p + epsilon * gnew / 2   # make half-step in p
        logPnew = target(thetanew)   # find new value of H
        Hnew = p.T @ p / 2 - logPnew
        dH = H - Hnew    # Decide whether to accept
        if np.log(npr.rand()) < dH:
            g = gnew
            theta = thetanew
            logP = logPnew
            accepts += 1
        thetas = jax.ops.index_update(thetas, jax.ops.index[m,:], theta)
    print('Acceptance rate:', accepts/M)
    return thetas, leapfrog_chain
 
# def sphere(theta):
#     return -20*(np.sqrt(np.sum(theta**2))-10)**2

xs = np.linspace(-5, 5, 100)
ys = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(xs, ys)
Z = jax.vmap(jax.vmap(lambda x, y: np.exp(target(np.array((x, y)))), in_axes=(0, 0)), in_axes=(0, 0))(X, Y)
 
L = 50
samples, leapfrog_chain = hmc(np.array([1.0, 0.0]), 200, target, 0.1, L)
chain = samples
samples = samples[samples.shape[0]//2:]
plt.contour(X, Y, Z, 20)
plt.plot(samples[:,0], samples[:,1])
plt.show()

fig, ax = plt.subplots()
# circle.plot_density(problem, ax)
ax.contour(X, Y, Z, 20)
line = ax.plot([], [])[0]
sample_line = ax.plot((), (), linestyle='', marker='.')[0]
def animate(i):
    min_i = np.max((0, i - i % L))
    line.set_data(leapfrog_chain[min_i:i, 0], leapfrog_chain[min_i:i, 1])
    sample_index = int(i / L)
    sample_line.set_data(chain[:sample_index, 0], chain[:sample_index, 1])
    return [line, sample_line]

anim = animation.FuncAnimation(
    fig, animate, leapfrog_chain.shape[0], interval=20, blit=True
)
plt.show()

