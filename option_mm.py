import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.misc import derivative

def u(w):
	return np.log(w)

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

w = [5000]*K.shape[0]
subjective = [1.0/K.shape[0]]*K.shape[0]

it = 0
while True:
	call_mm = []
	put_mm = []
	p_i = []

	dw = []
	for i in range(0, K.shape[0]):
		dw.append(derivative(u, w[i], dx=1e-6))

	deno = 0
	for i in range(0, K.shape[0]):
		deno = deno + 1.0/K.shape[0] * dw[i]

	for i in range(0, K.shape[0]):
		p_i.append(1.0/K.shape[0]*dw[i]/deno)

	for i in range(0, K.shape[0]):
		call_mm.append(np.exp(-r)*sum(np.maximum(K-K[i], 0)*p_i))
		put_mm.append(np.exp(-r)*sum(np.maximum(K[i]-K, 0)*p_i))

	# positive: traders buy from mm; negative: traders sell to mm
	call_quant = [0]*K.shape[0]
	call_buy_sell = np.array(call_discrete)-np.array(call_mm)
	for i in range(0, K.shape[0]):	
		call_quant = call_quant + np.maximum(K-K[i], 0)*np.sign(call_buy_sell[i])

	put_quant = [0]*K.shape[0]
	put_buy_sell = np.array(put_discrete)-np.array(put_mm)
	for i in range(0, K.shape[0]):
		put_quant = put_quant + np.maximum(K[i]-K, 0)*np.sign(put_buy_sell[i])

	w = w + call_mm*np.sign(call_buy_sell) + put_mm*np.sign(put_buy_sell)-call_quant-put_quant
	it = it+1
	if it%100==0:
		print(it)
		print(p_i)
		print(1.0*hist/sum(hist))
	#pdb.set_trace()
#pdb.set_trace()