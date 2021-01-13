
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

import numpy as np
import numpy.random as npr

## We want to release sample from N(R_mean, x_nc_var-Var(R))
# The renyi divergence between two univariate normal is:
# \ln(\sigma_2/\sigma_1)+1/(2*(\alpha-1))\ln(\sigma_2^2/(\sigma^2)^*_\alpha)+\alpha/2*(\mu_1-\mu_2)^2/(\sigma^2)^*_\alpha)

def from_RDP_to_DP(eps_rdp, alpha, delta):
	return eps_rdp-np.log(delta)/(alpha-1)

from scipy.special import binom
def amplified_RDP(eps_rdp_list, alpha, q):
	"""
	epsilon list will begin with e(2)
	"""
	term0 = 1
	term1 = q**2*binom(alpha, 2)*min(4*(np.exp(eps_rdp_list[0])-1), np.exp(eps_rdp_list[0])*2)
	term2 = sum([q**j*binom(alpha, j)*np.exp((j-1)*eps_rdp_list[j-2])*2 for j in range(3,alpha+1)])
	return 1/(alpha-1)*np.log((term0+term1+term2))

def rd_approx(alpha, b):
	return 5/(2*b)+1/(2*(alpha-1))*np.log(2*b/(b-5*alpha))+2*alpha/(b-5*alpha)

def get_privacy_spent(b, N, T, max_alpha=10, delta=None):
	max_alpha = min(max_alpha,b//5)
	if delta == None : delta = 1/N
	min_eps = np.inf
	q = b/N
	for max_alpha_ in range(3, max_alpha):
		eps_alpha_list = [rd_approx(alpha, b) for alpha in range(2, max_alpha_+1)]
		amplified_eps = amplified_RDP(eps_alpha_list, max_alpha_, q)
		if not np.isfinite(amplified_eps):
			break
		total_eps = from_RDP_to_DP(T*amplified_eps, max_alpha_, delta)
		min_eps = min(min_eps, total_eps)
	return (min_eps, delta)

