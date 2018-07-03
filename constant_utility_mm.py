import pdb
import numpy as np
import matplotlib.pyplot as plt
import bisect
from scipy.stats import norm, entropy, lognorm
from scipy.misc import derivative
from scipy.optimize import fsolve

class ConstantUtilityMM:
	def __init__(self, c_strikes, c_bids, c_asks, stock_price, days_to_expiration):
		self.c_strikes = c_strikes/1000
		self.c_bids = c_bids
		self.c_asks = c_asks
		self.stock_price = stock_price
		self.days_to_expiration = days_to_expiration/252

	def u(self, w):
		return np.log(w)

	def v(self, w):
		#return -np.exp(-0.1*w)
		return 1.0/w

	def func(self, x, p_s, q, k):
		if all(x-q > 0) == False:
			pdb.set_trace()
		# assert(all(x-q > 0)), "wealth falls negative."
		return sum(p_s*np.array(self.u(x-q)))-k

	def solve_implicit_C(self, cost, p_s, q, delta_q, k):
		#factor = 1
		new_cost, d, ier, mesg = fsolve(self.func, np.max(q+delta_q)+k, args=(p_s, q+delta_q, k), full_output=True)
		assert(mesg=='The solution converged.'), "The solution is not converged!"
		# while mesg != 'The solution converged.':
		# 	print(mesg)
		# 	factor = factor * 0.1
		# 	delta_q = factor * delta_q
		# 	new_cost, d, ier, mesg = fsolve(self.func, cost, args=(p_s, q+delta_q, k), full_output=True)
		return new_cost, delta_q

	def mm(self):
		fl = min(self.c_strikes)
		cl = max(self.c_strikes)
		K = np.linspace(fl, cl, (cl-fl)/0.5+1)
		cost = 1e4
		w = [cost] * K.shape[0]
		# the vector of all quantities of shares held by traders
		q = [0] * K.shape[0]
		# prior (set to a year in the future)
		# expected rate of return, volatility
		mu, sigma = 0.12, 0.16
		dist = lognorm(s=sigma, scale=np.exp(np.log(self.stock_price)+(mu-sigma*sigma/2)))
		# dist = lognorm(s=sigma*np.sqrt(self.days_to_expiration), scale=np.exp(np.log(self.stock_price)+(mu-sigma*sigma/2)*self.days_to_expiration))
		bins = np.linspace(K[0], K[-1]+0.5, (K[-1]+0.5-K[0])/0.5+1)
		subjective = (dist.cdf(bins)[1:] - dist.cdf(K))/sum(dist.cdf(bins)[1:] - dist.cdf(K))	
		# subjective = [1.0 / K.shape[0]] * K.shape[0]
		# calculate constant utility
		U = sum(subjective * np.array(self.u(w)))
		P = []

		it = 0
		while len(P) < 2 or (P[-1] != P[-2]).any():
			w = np.array(cost) - q
			assert(all(w > 0)), "wealth is negative at iteration {}.".format(it)
			dw = []
			for i in range(0, K.shape[0]):
				dw.append(derivative(self.u, w[i], dx=1e-6))
			deno = sum(subjective*np.array(dw))
			p_i = subjective*np.array(dw)/deno
			P.append(p_i)

			if it % 500 == 0:
				print('The constant utility is {}'.format(sum(subjective*np.array(self.u(w)))))
				print('The minimum wealth at iteration {} is {}'.format(it, min(w)))
				print(p_i)

			for i in range(0, K.shape[0]):
				j = bisect.bisect_left(self.c_strikes, K[i], lo=0, hi=len(self.c_strikes))
				if K[i] == self.c_strikes[j]:
					call_bid = self.c_bids[j]
					call_ask = self.c_asks[j]
					new_cost, q_delta = self.solve_implicit_C(cost, subjective, q, np.maximum(K-K[i], 0), U)
					call_mm_buy = new_cost - cost
					if np.sign(call_mm_buy)>=0 and np.absolute(np.absolute(call_mm_buy)-call_bid)>=0.01 \
						and np.absolute(call_mm_buy) < call_bid:
						q = q + q_delta
						cost = new_cost
					new_cost, q_delta = self.solve_implicit_C(cost, subjective, q, -np.maximum(K-K[i], 0), U)
					call_mm_sell = new_cost - cost
					if np.sign(call_mm_sell)<=0 and np.absolute(np.absolute(call_mm_sell)-call_ask)>=0.01 \
						and np.absolute(call_mm_sell) > call_ask:
						q = q + q_delta
						cost = new_cost
			it = it+1
		return K, P[-1]
	


