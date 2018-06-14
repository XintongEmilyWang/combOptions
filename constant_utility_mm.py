import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
from scipy.misc import derivative
from scipy.optimize import fsolve

class ConstantUtilityMM:
	def __init__(self, c_strikes, c_bids, c_asks, p_strikes, p_bids, p_asks):
		self.c_strikes = c_strikes
		self.c_bids = c_bids
		self.c_asks = c_asks
		self.p_strikes = p_strikes
		self.p_bids = p_bids
		self.p_asks = p_asks

	def u(w):
		#return -np.exp(-0.1*np.array(w))
		#return 1.0/np.array(w)
		return np.log(w)

	def v(w):
		#return -np.exp(-0.1*w)
		return 1.0/w

	def func(x, p_s, q, k):
		return sum(p_s*np.array(u(x-q)))-k

	def mm(self):
		pdb.set_trace()
		fl = min(min(self.c_strikes), min(self.p_strikes))
		cl = max(max(self.c_strikes), max(self.p_strikes))
		K = np.linspace(fl, cl, (cl-fl)/500+1)
		pdb.set_trace()

		cost = 10000
		w = [cost] * K.shape[0]
		# the vector of all quantities of shares held by traders
		q = [0] * K.shape[0]
		subjective = [1.0 / K.shape[0]] * K.shape[0]
		U = sum(subjective * np.array(u(w)))
		numIt = 10000
		loss = [0] * numIt
		P = []

		it = 0
		while it < numIt:
			w = np.array(cost) - q
			assert(w.all() > 0), "wealth is zero at iteration {}".format(it)
			dw = []
			for i in range(0, K.shape[0]):
				dw.append(derivative(u, w[i], dx=1e-6))
			deno = sum(subjective*np.array(dw))
			p_i = subjective*np.array(dw)/deno
			loss[it] = entropy(1.0*hist/sum(hist), p_i)
			P.append(p_i)

			if it % 50 == 0:
				print('The constant utility is {}'.format(sum(subjective*np.array(u(w)))))
				print('For iteration {}, the loss is {}'.format(it, loss[it]))
				print(p_i)

			for i in range(0, K.shape[0]):
				# call
				call_mm_buy = fsolve(func, cost, args=(subjective, q+np.maximum(K-K[i], 0), U))-cost
				call_mm_sell = fsolve(func, cost, args=(subjective, q-np.maximum(K-K[i], 0), U))-cost
				# TO-DO: could this happen at the sametime??
				if call_discrete[i] > np.absolute(call_mm_buy):
					q = q+np.maximum(K-K[i], 0)
					cost = cost+call_mm_buy
				if call_discrete[i] < np.absolute(call_mm_sell):
					q = q-np.maximum(K-K[i], 0)
					cost = cost+call_mm_sell
				# put
				put_mm_buy = fsolve(func, cost, args=(subjective, q+np.maximum(K[i]-K, 0), U))-cost
				put_mm_sell = fsolve(func, cost, args=(subjective, q-np.maximum(K[i]-K, 0), U))-cost	
				if put_discrete[i] > np.absolute(put_mm_buy):
					q = q+np.maximum(K[i]-K, 0)
					cost = cost+put_mm_buy
				if put_discrete[i] < np.absolute(put_mm_sell):
					q = q-np.maximum(K[i]-K, 0)
					cost = cost+put_mm_sell
			it = it+1

		# np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.4f}'.format})
		# print('The minimum loss is {}, reached at {} iteration.'.format(np.min(loss), np.argmin(loss)))
		# fig1 = plt.figure()
		# plt.plot(K, 1.0*hist/sum(hist), 'ro')
		# plt.plot(K, P[np.argmin(loss)], 'bx')
		# fig2 = plt.figure()
		# plt.plot(loss)
		pdb.set_trace()
	


