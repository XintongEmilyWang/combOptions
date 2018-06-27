import pdb
import numpy as np
import matplotlib.pyplot as plt
import bisect
from scipy.stats import norm, entropy
from scipy.misc import derivative
from scipy.optimize import fsolve

class ConstantUtilityMM:
	def __init__(self, c_strikes, c_bids, c_asks):
		self.c_strikes = c_strikes/1000
		self.c_bids = c_bids
		self.c_asks = c_asks
		# self.p_strikes = p_strikes/1000
		# self.p_bids = p_bids
		# self.p_asks = p_asks

	def u(self, w):
		#return -np.exp(-0.1*np.array(w))
		#return 1.0/np.array(w)
		return np.log(w)

	def v(self, w):
		#return -np.exp(-0.1*w)
		return 1.0/w

	def func(self, x, p_s, q, k):
		#assert(all(x-q > 0)), "wealth falls negative."
		return sum(p_s*np.array(self.u(x-q)))-k

	def solve_implicit_C(self, cost, p_s, q, delta_q, k):
		factor = 1
		new_cost, d, ier, mesg = fsolve(self.func, cost, args=(p_s, q+delta_q, k), full_output=True)
		# print(mesg)
		while mesg != 'The solution converged.':
			factor = factor * 0.1
			delta_q = factor * delta_q
			new_cost, d, ier, mesg = fsolve(self.func, cost, args=(p_s, q+delta_q, k), full_output=True)
		return new_cost, delta_q, factor

	def mm(self):
		fl = min(self.c_strikes)
		cl = max(self.c_strikes)
		K = np.linspace(fl, cl, (cl-fl)/0.5+1)

		cost = 1e4
		w = [cost] * K.shape[0]
		# the vector of all quantities of shares held by traders
		q = [0] * K.shape[0]
		subjective = [1.0 / K.shape[0]] * K.shape[0]
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

			if it % 100 == 0:
				print('The constant utility is {}'.format(sum(subjective*np.array(self.u(w)))))
				print('The minimum wealth at iteration {} is {}'.format(it, min(w)))
				print(p_i)

			# assert(self.c_strikes.shape[0] == self.p_strikes.shape[0]), "put call strike sizes are different."

			for i in range(0, K.shape[0]):
				# call
				j = bisect.bisect_left(self.c_strikes, K[i], lo=0, hi=len(self.c_strikes))
				if K[i] == self.c_strikes[j]:
					call_bid = self.c_bids[j]
					call_ask = self.c_asks[j]
					new_cost, q_delta, factor = self.solve_implicit_C(cost, subjective, q, np.maximum(K-K[i], 0), U)
					call_mm_buy = new_cost - cost
					if np.sign(call_mm_buy) >= 0 and np.absolute(np.absolute(call_mm_buy)-call_bid*factor) > 0.01 \
					and np.absolute(call_mm_buy) < call_bid*factor:
						q = q + q_delta
						cost = new_cost
					# pdb.set_trace()
					new_cost, q_delta, factor = self.solve_implicit_C(cost, subjective, q, -np.maximum(K-K[i], 0), U)
					call_mm_sell = new_cost - cost
					# pdb.set_trace()
					if np.sign(call_mm_sell) <= 0 and np.absolute(np.absolute(call_mm_sell)-call_ask*factor) > 0.01 \
					and np.absolute(call_mm_sell) > call_ask*factor:
						q = q + q_delta
						cost = new_cost
					# pdb.set_trace()
						# the constraint of neighbor strike prices is not helpful	
						# 	call_bid = self.c_asks[j]
						# 	call_ask = self.c_bids[j-1]
				# put
				# k = bisect.bisect_left(self.p_strikes, K[i], lo=0, hi=len(self.p_strikes))
				# if K[i] == self.p_strikes[k]:
				# 	put_bid = self.p_bids[k]
				# 	put_ask = self.p_asks[k]
				# 	new_cost, q_delta, factor = self.solve_implicit_C(cost, subjective, q, np.maximum(K[i]-K, 0), U)
				# 	put_mm_buy = new_cost - cost
				# 	if np.sign(put_mm_buy) >= 0 and np.absolute(np.absolute(put_mm_buy)-put_bid*factor) > 0.01 \
				# 	and np.absolute(put_mm_buy) < put_bid*factor:
				# 		q = q + q_delta
				# 		cost = new_cost
				# 	# pdb.set_trace()
				# 	new_cost, q_delta, factor = self.solve_implicit_C(cost, subjective, q, -np.maximum(K[i]-K, 0), U)
				# 	put_mm_sell = new_cost - cost
				# 	if np.sign(put_mm_sell) <= 0 and np.absolute(np.absolute(put_mm_sell)-put_ask*factor) > 0.01 \
				# 	and np.absolute(put_mm_sell) > put_ask*factor:
				# 		q = q + q_delta
				# 		cost = new_cost
					# pdb.set_trace()
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
		return K, P[-1]
	


