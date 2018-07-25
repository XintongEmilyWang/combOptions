import pdb
import sys
import os
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

input_dir = sys.argv[1]
price_date = sys.argv[2]
buy_or_sell = sys.argv[3]
coeff1 = int(sys.argv[4])
st1 = sys.argv[5]
coeff2 = int(sys.argv[6])
st2 = sys.argv[7]
strike = int(sys.argv[8])
price = float(sys.argv[9])
expiration_date = int(sys.argv[10])
comb_type = sys.argv[11]

opt1_stats_filename = os.path.join(input_dir, st1+"_"+price_date+".xlsx")
opt1_df = pd.read_excel(opt1_stats_filename)
opt2_stats_filename = os.path.join(input_dir, st2+"_"+price_date+".xlsx")
opt2_df = pd.read_excel(opt2_stats_filename)

assert(np.array(opt1_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"
assert(np.array(opt2_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"

# opt1_df = opt1_df[(opt1_df['Expiration Date of the Option']==expiration_date) & (opt1_df['C=Call, P=Put']=='C') & (opt1_df['Highest Closing Bid Across All Exchanges'] > 0)]
# opt2_df = opt2_df[(opt2_df['Expiration Date of the Option']==expiration_date) & (opt2_df['C=Call, P=Put']=='C') & (opt2_df['Highest Closing Bid Across All Exchanges'] > 0)]
opt1_df = opt1_df[(opt1_df['Expiration Date of the Option']==expiration_date) & (opt1_df['Highest Closing Bid Across All Exchanges'] > 0)]
opt2_df = opt2_df[(opt2_df['Expiration Date of the Option']==expiration_date) & (opt2_df['Highest Closing Bid Across All Exchanges'] > 0)]
opt1_df['Strike Price of the Option Times 1000'] = opt1_df['Strike Price of the Option Times 1000'].astype(int)
opt1_df['Highest Closing Bid Across All Exchanges'] = opt1_df['Highest Closing Bid Across All Exchanges'].astype(float)
opt1_df['Lowest  Closing Ask Across All Exchanges'] = opt1_df['Lowest  Closing Ask Across All Exchanges'].astype(float)
opt2_df['Strike Price of the Option Times 1000'] = opt2_df['Strike Price of the Option Times 1000'].astype(int)
opt2_df['Highest Closing Bid Across All Exchanges'] = opt2_df['Highest Closing Bid Across All Exchanges'].astype(float)
opt2_df['Lowest  Closing Ask Across All Exchanges'] = opt2_df['Lowest  Closing Ask Across All Exchanges'].astype(float)
opt1_df.sort_values(by = ['Strike Price of the Option Times 1000'], inplace = True)
opt2_df.sort_values(by = ['Strike Price of the Option Times 1000'], inplace = True)

if buy_or_sell == 'buy':
	# The exchange would sell an item with intrinsic low value at a higher price and cover with buying an item with intrinsic high value at a lower price.
	opt1_df = opt1_df[opt1_df['C=Call, P=Put']=='C']
	opt2_df = opt2_df[opt2_df['C=Call, P=Put']=='C']
	opt1_bids = np.array(opt1_df['Highest Closing Bid Across All Exchanges'])
	opt2_bids = np.array(opt2_df['Highest Closing Bid Across All Exchanges'])
	bids = np.concatenate([opt1_bids, opt2_bids])
	opt1_asks = np.array(opt1_df['Lowest  Closing Ask Across All Exchanges'])
	opt2_asks = np.array(opt2_df['Lowest  Closing Ask Across All Exchanges'])
	asks = np.concatenate([opt1_asks, opt2_asks])
	opt1_strikes = np.array(opt1_df['Strike Price of the Option Times 1000'])
	opt2_strikes = np.array(opt2_df['Strike Price of the Option Times 1000'])
	strikes = np.concatenate([opt1_strikes, opt2_strikes])

	# coefficients of the objective function
	c = matrix(np.concatenate([asks, [-1*price]]))
	# constant constraints of each inequality
	b = matrix(np.concatenate([np.concatenate([np.zeros(3+len(asks)), np.ones(len(asks))*999]), [0, 1]]))
	# coeffecient of each variable for each inequality
	A = np.zeros((len(asks)+1, (len(asks)+1)*2+3))
	for i in range(0, len(asks)):
		temp = np.zeros((len(asks)+1)*2+3)
		if i < len(opt1_asks):
			temp[0] = -1
			temp[1] = 0
		else:
			temp[0] = 0
			temp[1] = -1
		temp[2] = strikes[i]
		temp[3+i] = -1
		temp[3+len(asks)+i] = 1
		A[i, :] = temp
	A[-1, :] =  np.concatenate([[coeff1, coeff2, -1*strike], np.zeros((len(asks)+1)*2)])
	A[-1, -1] = 1
	A[-1, -2] = -1
	M = matrix(A.transpose())
	sol=solvers.lp(c,M,b,verbose=True)
	frac = np.array(sol['x'])

	accept_frac = float("{0:.2f}".format(frac[-1, 0]))
	print('The exchange sells by accepting {} fraction of the combinatorial bid order on call option of {}{}+{}{} with strike price {} at bid price {}'.format(accept_frac, coeff1, st1, coeff2, st2, strike, price))
	print('The exchange will cover the {} fraction of the combinatorial bid order with...'.format(accept_frac))
	profit = accept_frac*price
	for i in range(0, len(frac[:-1])):
		if frac[i, 0] > 1e-2:
			if i < len(opt1_asks):
				profit = profit-frac[i, 0]*asks[i]
				print("Buy {} fraction of call option on {} at strike {} with ask price {}".format(float("{0:.4f}".format(frac[i, 0])), st1, strikes[i], asks[i]))
			else: 
				profit = profit-frac[i, 0]*asks[i]
				print("Buy {} fraction of call option on {} at strike {} with ask price {}".format(float("{0:.4f}".format(frac[i, 0])), st2, strikes[i], asks[i]))
	print('The maximized revenue is {}.'.format(profit))
else: 
	# The exchange would buy an item with intrinsic high value at a lower price and cover with selling an item with intrinsic lower value at a higher price.
	opt1_df = opt1_df[opt1_df['C=Call, P=Put']=='C']
	opt2_df = opt2_df[opt2_df['C=Call, P=Put']=='P']
	opt1_bids = np.array(opt1_df['Highest Closing Bid Across All Exchanges'])
	opt2_asks = np.array(opt2_df['Lowest  Closing Ask Across All Exchanges'])
	opt1_strikes = np.array(opt1_df['Strike Price of the Option Times 1000'])
	opt2_strikes = np.array(opt2_df['Strike Price of the Option Times 1000'])
	strikes = np.concatenate([opt1_strikes, opt2_strikes])

	c = matrix(np.concatenate([np.concatenate([-1*opt1_bids, opt2_asks]), [price]]))
	# b = matrix(np.concatenate([np.concatenate([np.concatenate([np.zeros(3+len(bids)), np.ones(len(bids))*999]), -0.0001*np.zeros(len(bids))]), [0, 1]]))
	b = matrix(np.concatenate([np.concatenate([np.zeros(3+len(strikes)), np.ones(len(strikes))*999]), [0, 1]]))
	A = np.zeros((len(strikes)+1, len(strikes)*2+5))
	for i in range(0, len(strikes)):
		temp = np.zeros(len(strikes)*2+5)
		if i < len(opt1_bids):
			temp[0] = 1
			temp[1] = 0
		else:
			temp[0] = 0
			temp[1] = 1
		temp[2] = -strikes[i]
		temp[3+i] = -1
		temp[3+len(strikes)+i] = 1
		# temp[3+2*len(bids)+i] = -1*bids[i]
		A[i, :] = temp
	A[-1, :] =  np.concatenate([[-coeff1, -coeff2, strike], np.zeros(len(strikes)*2+2)])
	A[-1, -1] = 1
	A[-1, -2] = -1
	M = matrix(A.transpose())
	sol=solvers.lp(c,M,b, verbose = True)
	frac = np.array(sol['x'])

	accept_frac = float("{0:.2f}".format(frac[-1, 0]))
	print('The exchange buys by accepting {} fraction of the combinatorial ask order on call option of {}{}+{}{} with strike price {} at ask price {}'.format(accept_frac, coeff1, st1, coeff2, st2, strike, price))
	print('The exchange will cover the {} fraction of the combinatorial ask order with...'.format(accept_frac))
	profit = -1*accept_frac*price
	for i in range(0, len(frac[:-1])):
		if frac[i, 0] > 1e-2:
			if i < len(opt1_bids):
				profit = profit+frac[i, 0]*opt1_bids[i]
				print("Sell {} fraction of call option on {} at strike {} with bid price {}".format(float("{0:.4f}".format(frac[i, 0])), st1, strikes[i], opt1_bids[i]))
			else: 
				profit = profit-frac[i, 0]*opt2_asks[i-len(opt1_bids)]
				print("Buy {} fraction of put option on {} at strike {} with ask price {}".format(float("{0:.4f}".format(frac[i, 0])), st2, strikes[i], opt2_asks[i-len(opt1_bids)]))
	print('The maximized revenue is {}.'.format(profit))


