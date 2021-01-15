
# MIT License

# Copyright (c) 2019 DPBayes

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The code has been modified by Ossi Räisä
import numpy as np
import numpy.random as npr
from scipy.special import expit as logistic
import dp_mcmc_module.X_corr as X_corr

def run_dp_Barker(problem, T, prop_var, theta0, temp_scale, x_corr_df, n_points, batch_size=100, verbose=True):
	"""
	T : number of iterations
	theta_0 : the starting value for chain
	"""
	data = problem.data
	N, data_dim = data.shape
	d = theta0.size
	privacy_pars = {
		"noise_scale": np.sqrt(2),
		"clip": [0, 0.99*np.sqrt(batch_size)/temp_scale/N]
	}
	privacy = False
	if privacy_pars['noise_scale']>0:
		privacy = True
	if privacy : clip_bounds = privacy_pars['clip']
	else : clip_bounds = [0, np.inf]
	theta_chain = np.zeros((T+1,d)) # (alpha, beta)
	theta_chain[0,:] = theta0 # Initialize chain to given point

	clip_count = np.zeros(T + 1)
	accepts = 0

	# Run the chain
	for i in range(1,T+1):
		# draw minibatch, use fixed batch size without regard to batch var
		X_batch_inds = np.random.choice(N, batch_size, replace=False)
		X_batch = data[X_batch_inds, :]
		# proposals from standard Gaussians
		proposal = theta_chain[i-1,:] + np.random.normal(0, np.sqrt(prop_var), d)
		# Compute log likelihoods for current and proposed value of the chain
		log_likelihood_theta_prime = problem.log_likelihood_no_sum(proposal, X_batch)
		log_likelihood_theta = problem.log_likelihood_no_sum(theta_chain[i-1], X_batch)
		# Compute log prior probabilities for current and proposed value of the chain
		log_prior_theta_prime = problem.log_prior(proposal)
		log_prior_theta = problem.log_prior(theta_chain[i-1])
		# Compute the log likelihood ratios
		ratio = log_likelihood_theta_prime - log_likelihood_theta
		sign = np.sign(ratio)
		# Clip the ratios
		if privacy : R = np.clip(np.abs(ratio), a_min=clip_bounds[0], a_max=clip_bounds[1])*sign
		else : R = ratio
		clip_count[i] = np.sum(np.abs(ratio) > clip_bounds[1])
		# Compute mean and sample variance of log-likelihoods
		r = R.mean()
		s2 = R.var()
		# Compute \Theta^*(\theta', \theta)
		logp_ratio = N * temp_scale * r + (log_prior_theta_prime - log_prior_theta)
		# Compute scaled batch var
		batch_var = s2 * ((N * temp_scale)**2 / batch_size)
		if privacy:
			if batch_var <= 1:
				#normal_noise = npr.randn(1)*np.sqrt(1-batch_var)
				normal_noise = npr.randn(1)*np.sqrt(2-batch_var) # should this be fixed or with given norma std?
				x_corr = X_corr.sample_from_mix(x_corr_df, 1)[0]
			else:
				print('Batch var > 1, exit')
				print(batch_var)
				break
		else:
			#normal_noise = 0 ## NOTE
			#x_corr = np.random.logistic(loc=0, scale=1) ## NOTE
			if batch_var <= 1:
				normal_noise = npr.randn(1)*np.sqrt(1.-batch_var)
				x_corr = X_corr.sample_from_mix(x_corr_df, 1)[0]
			else:
				print('Batch var > 1, exit')
				break

		acc =  logp_ratio + normal_noise + x_corr
		# acc = logp_ratio + np.random.logistic()

		if acc > 0:
			# accept
			theta_chain[i,:] = proposal
			accepts += 1
		else:
			# reject
			theta_chain[i,:] = theta_chain[i-1,:]
		if verbose and (i + 1) % 100 == 0:
			print("Iteration: {}".format(i + 1))
	return theta_chain, clip_count, accepts
