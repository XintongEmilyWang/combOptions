import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, entropy
from scipy.misc import derivative
from scipy.optimize import fsolve
import single_option_bounds

def u(w):
	#return -np.exp(-0.1*np.array(w))
	#return 1.0/np.array(w)
	return np.log(w)

def v(w):
	#return -np.exp(-0.1*w)
	return 1.0/w

def func(x, p_s, q, k):
	return sum(p_s*np.array(u(x-q)))-k

def mm(K, call_bids, call_asks, put_bids, put_asks):
	cost = 10000
	w = [cost] * K.shape[0]
	# the vector of all quantities of shares held by traders
	q = [0] * K.shape[0]
	subjective = [1.0 / K.shape[0]] * K.shape[0]
	U = sum(subjective * np.array(u(w)))
	numIt = 2000
	loss = []
	P = []

	it = 0
	while len(P) < 2 or (P[-1] != P[-2]).any():
		w = np.array(cost) - q
		assert(w.all() > 0), "wealth is zero at iteration {}".format(it)
		dw = []
		for i in range(0, K.shape[0]):
			dw.append(derivative(u, w[i], dx=1e-6))
		deno = sum(subjective*np.array(dw))
		p_i = subjective*np.array(dw)/deno
		loss.append(entropy(1.0*hist/sum(hist), p_i))
		P.append(p_i)

		if it % 1000 == 0:
			print('The constant utility is {}'.format(sum(subjective*np.array(u(w)))))
			print('For iteration {}, the loss is {}'.format(it, loss[-1]))
			print('The minimum wealth at iteration {} is {}'.format(it, min(w)))
			print(p_i)

		for i in range(0, K.shape[0]):
			# call
			call_mm_buy = fsolve(func, np.max(q+np.maximum(K-K[i], 0))+U, args=(subjective, q+np.maximum(K-K[i], 0), U))-cost
			if np.absolute(call_bids[i]-np.absolute(call_mm_buy))>0.005 and call_bids[i] > np.absolute(call_mm_buy):
				q = q+np.maximum(K-K[i], 0)
				cost = cost+call_mm_buy
			call_mm_sell = fsolve(func, np.max(q+np.maximum(K-K[i], 0))+U, args=(subjective, q-np.maximum(K-K[i], 0), U))-cost
			if np.absolute(call_asks[i]-np.absolute(call_mm_sell))>0.005 and call_asks[i] < np.absolute(call_mm_sell):
				q = q-np.maximum(K-K[i], 0)
				cost = cost+call_mm_sell
			# put
			put_mm_buy = fsolve(func, np.max(q+np.maximum(K-K[i], 0))+U, args=(subjective, q+np.maximum(K[i]-K, 0), U))-cost
			if np.absolute(put_bids[i]-np.absolute(put_mm_buy))>0.005 and put_bids[i] > np.absolute(put_mm_buy):
				q = q+np.maximum(K[i]-K, 0)
				cost = cost+put_mm_buy
			put_mm_sell = fsolve(func, np.max(q+np.maximum(K-K[i], 0))+U, args=(subjective, q-np.maximum(K[i]-K, 0), U))-cost	
			if np.absolute(put_asks[i]-np.absolute(put_mm_sell))>0.005 and put_asks[i] < np.absolute(put_mm_sell):
				q = q-np.maximum(K[i]-K, 0)
				cost = cost+put_mm_sell
		it = it+1
	np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.4f}'.format})
	print('The loss is {}, reached at {} iteration.'.format(loss[-1], it))
	return loss[-1]


r, mu, sigma, s0 = 0.08, 0.15, 0.3, 50
in_the_money_prop = 0.05
around_the_money_prop = 0.08
out_the_money_prop = 0.15
num_sim = 10
loss = np.zeros([3, num_sim])

it = 0
while it < num_sim:
	st = np.random.lognormal(np.log(s0)+(mu-sigma*sigma/2), sigma, 10000)
	fl = np.floor(np.min(st)/5)*5
	cl = np.ceil(np.max(st)/5)*5
	hist, bin_edges = np.histogram(st, bins = np.linspace(fl, cl, (cl-fl)/5+1))
	# plt.hist(st, bins = np.linspace(fl, cl, (cl-fl)/5+1))
	# plt.show()

	K = bin_edges[:-1]
	# call = []
	# put = []
	# for k in K:
	# 	d1 = (np.log(s0/k)+(r+sigma*sigma/2))/sigma
	# 	d2 = d1-sigma
	# 	call.append(np.exp(-r)*(s0*norm.cdf(d1)*np.exp(r)-k*norm.cdf(d2)))
	# 	put.append(k*np.exp(-r)*norm.cdf(-d2)-s0*norm.cdf(-d1))

	# ground truth
	call_discrete = []
	put_discrete = []
	for k in K:
		call_discrete.append(sum(np.maximum(K-k, 0)*hist/hist.sum()))
		put_discrete.append(sum(np.maximum(k-K, 0)*hist/hist.sum()))

	# bid and ask spread with gaussian noise
	call_bids = []
	call_asks = []
	put_bids = []
	put_asks = []
	for i in range(0, len(K)):
		call_bid = 1000
		call_ask = 0
		put_bid = 1000
		put_ask = 0
		if K[i] < 45:
			while call_bid > call_ask:
				call_bid = call_discrete[i] - in_the_money_prop*call_discrete[i]+np.random.normal(0, in_the_money_prop*call_discrete[i]*0.25, 1)
				call_ask = call_discrete[i] + in_the_money_prop*call_discrete[i]+np.random.normal(0, in_the_money_prop*call_discrete[i]*0.25, 1)
			while put_bid > put_ask:
				if put_discrete[i] == 0:
					put_bid = [0]
					put_ask = [0]
				else:
					put_bid = put_discrete[i] - out_the_money_prop*put_discrete[i]+np.random.normal(0, out_the_money_prop*put_discrete[i]*0.25, 1)
					put_ask = put_discrete[i] + out_the_money_prop*put_discrete[i]+np.random.normal(0, out_the_money_prop*put_discrete[i]*0.25, 1)
		elif K[i] >= 45 and K[i] <=55:
			while call_bid > call_ask:
				call_bid = call_discrete[i] - around_the_money_prop*call_discrete[i]+np.random.normal(0, around_the_money_prop*call_discrete[i]*0.25, 1)
				call_ask = call_discrete[i] + around_the_money_prop*call_discrete[i]+np.random.normal(0, around_the_money_prop*call_discrete[i]*0.25, 1)
			while put_bid > put_ask:
				put_bid = put_discrete[i] - around_the_money_prop*put_discrete[i]+np.random.normal(0, around_the_money_prop*put_discrete[i]*0.25, 1)
				put_ask = put_discrete[i] + around_the_money_prop*put_discrete[i]+np.random.normal(0, around_the_money_prop*put_discrete[i]*0.25, 1)
		else:
			while call_bid > call_ask:
				if call_discrete[i] == 0:
					call_bid = [0]
					call_ask = [0]
				else: 
					call_bid = call_discrete[i] - out_the_money_prop*call_discrete[i]+np.random.normal(0, out_the_money_prop*call_discrete[i]*0.25, 1)
					call_ask = call_discrete[i] + out_the_money_prop*call_discrete[i]+np.random.normal(0, out_the_money_prop*call_discrete[i]*0.25, 1)
			while put_bid > put_ask:
				put_bid = put_discrete[i] - in_the_money_prop*put_discrete[i]+np.random.normal(0, in_the_money_prop*put_discrete[i]*0.25, 1)
				put_ask = put_discrete[i] + in_the_money_prop*put_discrete[i]+np.random.normal(0, in_the_money_prop*put_discrete[i]*0.25, 1)
		
		call_bids.append(call_bid[0])
		call_asks.append(call_ask[0])
		put_bids.append(put_bid[0])
		put_asks.append(put_ask[0])

	# tighten the spread
	d = {'C=Call, P=Put': np.concatenate((['C']*len(K), ['P']*len(K)), axis=0), \
		'Unit': np.concatenate(([1]*len(K), [-1]*len(K)), axis=0), \
		'Strike Price of the Option Times 1000': np.concatenate((K, -1*K), axis=0), \
		'Highest Closing Bid Across All Exchanges': np.concatenate((call_bids, put_bids), axis=0), \
		'Lowest  Closing Ask Across All Exchanges': np.concatenate((call_asks, put_asks), axis=0), \
		'Expiration Date of the Option': [252]*2*len(K)}
	opt_df = pd.DataFrame(data=d)
	spread_shrinker = single_option_bounds.SingleOptionBounds(st='S', opt_df=opt_df, strike_array = K, days_to_expiration=252)
	arbitrage, strikes, call_shrinked_bids, call_shrinked_asks, put_shrinked_bids, put_shrinked_asks = spread_shrinker.tighten_spread()

	# print(arbitrage)

	if arbitrage == 0:
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~At iteration {}.~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(it))
		print('loss at ground truth.')
		gt_loss = mm(K, call_discrete, call_discrete, put_discrete, put_discrete)
		print('loss at raw quotes.')
		raw_loss = mm(K, call_bids, call_asks, put_bids, put_asks)
		print('loss at tightened quotes.')
		lp_loss = mm(K, call_shrinked_bids, call_shrinked_asks, put_shrinked_bids, put_shrinked_asks)
		loss[0, it] = gt_loss
		loss[1, it] = raw_loss
		loss[2, it] = lp_loss
		it = it+1

np.save('loss.npy', loss)
pdb.set_trace()





# fig1 = plt.figure()
# plt.plot(K, 1.0*hist/sum(hist), 'ro')
# plt.plot(K, P[np.argmin(loss)], 'bx')
# fig2 = plt.figure()
# plt.plot(loss)
# pdb.set_trace()