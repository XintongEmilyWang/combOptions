import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
from scipy.misc import derivative
from scipy.optimize import fsolve

def u(w):
	#return -np.exp(-0.1*w)
	return np.log(w)

def v(w):
	#return -np.exp(-0.1*w)
	return 1.0/w

def func(x, p_s, q, k):
	return sum(p_s*np.array(u(x-q)))-k

r, mu, sigma, s0 = 0.08, 0.15, 0.3, 50
np.random.seed(101)
st = np.random.lognormal(np.log(s0)+(mu-sigma*sigma/2), sigma, 10000)
fl = np.floor(np.min(st)/5)*5
cl = np.ceil(np.max(st)/5)*5
hist, bin_edges = np.histogram(st, bins = np.linspace(fl, cl, (cl-fl)/5+1))
#plt.hist(st, bins = np.linspace(fl, cl, (cl-fl)/5+1))
#plt.show()

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

w = [10000]*K.shape[0]
subjective = [1.0/K.shape[0]]*K.shape[0]
U = sum(subjective*np.array(u(w)))
it = 0

while it <= 2000:
	call_mm = []
	put_mm = []
	p_i = []

	dw = []
	for i in range(0, K.shape[0]):
		dw.append(derivative(u, w[i], dx=1e-6))
	deno = sum(1.0/K.shape[0]*np.array(dw))
	p_i = 1.0/K.shape[0]*np.array(dw)/deno
	#cost = fsolve(func, 1000, )

	#KL divergence
	loss = entropy(1.0*hist/sum(hist), p_i)
		
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

	if it%50==0:
		print('The constant utility is {}'.format(sum(subjective*np.array(u(w)))))
		print('For iteration {}, the loss is {}'.format(it, loss))
		print(p_i)
		# print(call_quant+put_quant)
		# print(w)

	# TO-DO: cost should be the implicit solution of the constant utility function
	w = w + call_mm*np.sign(call_buy_sell) + put_mm*np.sign(put_buy_sell)-\
		call_quant-put_quant
	it = it+1
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.4f}'.format})
plt.plot(K, 1.0*hist/sum(hist), 'ro')
plt.plot(K, p_i, 'bx')
plt.show()
# pdb.set_trace()