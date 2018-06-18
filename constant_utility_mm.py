import pdb
import numpy as np
import matplotlib.pyplot as plt
import bisect
from scipy.stats import norm, entropy
from scipy.misc import derivative
from scipy.optimize import fsolve

class ConstantUtilityMM:
	def __init__(self, c_strikes, c_bids, c_asks, p_strikes, p_bids, p_asks):
		self.c_strikes = c_strikes/1000
		self.c_bids = c_bids
		self.c_asks = c_asks
		self.p_strikes = p_strikes/1000
		self.p_bids = p_bids
		self.p_asks = p_asks

	def u(self, w):
		#return -np.exp(-0.1*np.array(w))
		#return 1.0/np.array(w)
		return np.log(w)

	def v(self, w):
		#return -np.exp(-0.1*w)
		return 1.0/w

	def func(self, x, p_s, q, k):
		return sum(p_s*np.array(self.u(x-q)))-k

	def mm(self):
		fl = min(min(self.c_strikes), min(self.p_strikes))
		cl = max(max(self.c_strikes), max(self.p_strikes))
		K = np.linspace(fl, cl, (cl-fl)/0.5+1)
		pdb.set_trace()

		cost = 1e5
		w = [cost] * K.shape[0]
		# the vector of all quantities of shares held by traders
		q = [0] * K.shape[0]
		subjective = [1.0 / K.shape[0]] * K.shape[0]
		U = sum(subjective * np.array(self.u(w)))
		numIt = 10000
		P = []

		it = 0
		while it < numIt:
			w = np.array(cost) - q
			assert(w.all() > 0), "wealth is zero at iteration {}.".format(it)
			dw = []
			for i in range(0, K.shape[0]):
				dw.append(derivative(self.u, w[i], dx=1e-6))
			deno = sum(subjective*np.array(dw))
			p_i = subjective*np.array(dw)/deno
			P.append(p_i)

			if it % 50 == 0:
				print('The constant utility is {}'.format(sum(subjective*np.array(self.u(w)))))
				print('The minimum wealth at iteration {} is {}'.format(it, min(w)))
				print(p_i)

			assert(self.c_strikes.shape[0] == self.p_strikes.shape[0]), "put call strike sizes are different."

			for i in range(0, K.shape[0]):
				# call
				j = bisect.bisect_left(self.c_strikes, K[i], lo=0, hi=len(self.c_strikes))
				if K[i] == self.c_strikes[j]:
					call_bid = self.c_bids[j]
					call_ask = self.c_asks[j]
					call_mm_buy = fsolve(self.func, cost, args=(subjective, q+np.maximum(K-K[i], 0), U))-cost
					if np.sign(call_mm_buy)>=0 and np.absolute(np.absolute(call_mm_buy)-call_bid)>0.01 and np.absolute(call_mm_buy) < call_bid:
						q = q+np.maximum(K-K[i], 0)
						cost = cost+call_mm_buy
					call_mm_sell = fsolve(self.func, cost, args=(subjective, q-np.maximum(K-K[i], 0), U))-cost
					if np.sign(call_mm_sell)<=0 and np.absolute(np.absolute(call_mm_sell)-call_ask)>0.01 and np.absolute(call_mm_sell) > call_ask:
						q = q-np.maximum(K-K[i], 0)
						cost = cost+call_mm_sell
						# the constraint of neighbor strike prices is not helpful	
						# 	call_bid = self.c_asks[j]
						# 	call_ask = self.c_bids[j-1]
				# put
				k = bisect.bisect_left(self.p_strikes, K[i], lo=0, hi=len(self.p_strikes))
				if K[i] == self.p_strikes[k]:
					put_bid = self.p_bids[k]
					put_ask = self.p_asks[k]
					put_mm_buy = fsolve(self.func, cost, args=(subjective, q+np.maximum(K[i]-K, 0), U))-cost
					if np.sign(put_mm_buy)>=0 and np.absolute(np.absolute(put_mm_buy)-put_bid)>0.01 and np.absolute(put_mm_buy) < put_bid:
						q = q+np.maximum(K[i]-K, 0)
						cost = cost+put_mm_buy
					put_mm_sell = fsolve(self.func, cost, args=(subjective, q-np.maximum(K[i]-K, 0), U))-cost	
					if np.sign(put_mm_sell)<=0 and np.absolute(np.absolute(put_mm_sell)-put_ask)>0.01 and np.absolute(put_mm_sell) > put_ask:
						q = q-np.maximum(K[i]-K, 0)
						cost = cost+put_mm_sell
						# 	put_bid = self.p_asks[k-1]
						# 	put_ask = self.p_bids[k]
			it = it+1

		# np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.4f}'.format})
		# print('The minimum loss is {}, reached at {} iteration.'.format(np.min(loss), np.argmin(loss)))
		# fig1 = plt.figure()
		# plt.plot(K, 1.0*hist/sum(hist), 'ro')
		# plt.plot(K, P[np.argmin(loss)], 'bx')
		# fig2 = plt.figure()
		# plt.plot(loss)
		pdb.set_trace()
	


