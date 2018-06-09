import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.misc import derivative
r, mu, sigma, s0 = 0.08, 0.15, 0.3, 50
# one year span
st = np.random.lognormal(np.log(s0)+(mu-sigma*sigma/2), sigma, 10000)
fl = np.floor(np.min(st)/5)*5
cl = np.ceil(np.max(st)/5)*5
hist, bin_edges = np.histogram(st, bins = np.linspace(fl, cl, (cl-fl)/5+1))
#plt.hist(st, bins = np.linspace(fl, cl, (cl-fl)/5+1))
#plt.show()
# strike prices considered
K = bin_edges[:-1]
call = []
put = []
for k in K:
	d1 = (np.log(s0/k)+(r+sigma*sigma/2))/sigma
	d2 = d1-sigma
	call.append(np.exp(-r)*(s0*norm.cdf(d1)*np.exp(r)-k*norm.cdf(d2)))
	put.append(k*np.exp(-r)*norm.cdf(-d2)-s0*norm.cdf(-d1))

call_discrete = []
put_discrete = []
for k in K:
	call_discrete.append(np.exp(-r)*sum(np.maximum(K-k, 0)*hist/hist.sum()))
	put_discrete.append(np.exp(-r)*sum(np.maximum(k-K, 0)*hist/hist.sum()))

call_mm = []
put_mm = []
for k in K:
	call_mm.append(np.exp(-r)*sum(np.maximum(K-k, 0)*1/K.shape[0]))
	put_mm.append(np.exp(-r)*sum(np.maximum(k-K, 0)*1/K.shape[0]))

quant = [0]*K.shape[0]
buy_sell = np.array(call_discrete)-np.array(call_mm)
for i in range(0, K.shape[0]):	
	quant = quant + np.maximum(K-K[i], 0)*np.sign(buy_sell[i])
pdb.set_trace()
#def u(w):
#	return np.log(w)
#pdb.set_trace()