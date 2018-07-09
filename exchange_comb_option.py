import pdb
import sys
import os
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

input_dir = sys.argv[1]
price_date = sys.argv[2]
coeff1 = int(sys.argv[3])
st1 = sys.argv[4]
coeff2 = int(sys.argv[5])
st2 = sys.argv[6]
strike = int(sys.argv[7])
bid = float(sys.argv[8])
expiration_date = int(sys.argv[9])
comb_type = sys.argv[10]

opt1_stats_filename = os.path.join(input_dir, st1+"_"+price_date+".xlsx")
opt1_df = pd.read_excel(opt1_stats_filename)
opt2_stats_filename = os.path.join(input_dir, st2+"_"+price_date+".xlsx")
opt2_df = pd.read_excel(opt2_stats_filename)

assert(np.array(opt1_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"
assert(np.array(opt2_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"

opt1_df = opt1_df[(opt1_df['Expiration Date of the Option']==expiration_date) & (opt1_df['C=Call, P=Put']=='C')]
opt2_df = opt2_df[(opt2_df['Expiration Date of the Option']==expiration_date) & (opt2_df['C=Call, P=Put']=='C')]
opt1_df['Strike Price of the Option Times 1000'] = opt1_df['Strike Price of the Option Times 1000'].astype(int)
opt1_df['Highest Closing Bid Across All Exchanges'] = opt1_df['Highest Closing Bid Across All Exchanges'].astype(float)
opt1_df['Lowest  Closing Ask Across All Exchanges'] = opt1_df['Lowest  Closing Ask Across All Exchanges'].astype(float)
opt2_df['Strike Price of the Option Times 1000'] = opt2_df['Strike Price of the Option Times 1000'].astype(int)
opt2_df['Highest Closing Bid Across All Exchanges'] = opt2_df['Highest Closing Bid Across All Exchanges'].astype(float)
opt2_df['Lowest  Closing Ask Across All Exchanges'] = opt2_df['Lowest  Closing Ask Across All Exchanges'].astype(float)
opt1_df.sort_values(by = ['Strike Price of the Option Times 1000'], inplace = True)
opt2_df.sort_values(by = ['Strike Price of the Option Times 1000'], inplace = True)

opt1_asks = np.array(opt1_df['Lowest  Closing Ask Across All Exchanges'])
opt2_asks = np.array(opt2_df['Lowest  Closing Ask Across All Exchanges'])
asks = np.concatenate([opt1_asks, opt2_asks])
opt1_strikes = np.array(opt1_df['Strike Price of the Option Times 1000'])
opt2_strikes = np.array(opt2_df['Strike Price of the Option Times 1000'])
strikes = np.concatenate([opt1_strikes, opt2_strikes])

# coefficients of the objective function
c = matrix(np.concatenate([asks, [-1*bid]]))
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
	temp[3+len(opt1_asks)+i] = 1
	A[i, :] = temp
A[-1, :] =  np.concatenate([[coeff1, coeff2, -1*strike], np.zeros((len(asks)+1)*2)])
A[-1, -1] = 1
A[-1, -2] = -1
M = matrix(A.transpose())
sol=solvers.lp(c,M,b)
frac = np.array(sol['x'])

accept_frac = float("{0:.2f}".format(frac[-1, 0]))
print('Accept {} fraction of the combinatorial bid order on call option of {}{}+{}{} with strike price {} at bid price {}'.format(accept_frac, coeff1, st1, coeff2, st2, strike, bid))
print('The exchange will cover the {} fraction of the combinatorial bid order with...'.format(accept_frac))
profit = accept_frac*bid
for i in range(0, len(frac[:-1])):
	if frac[i, 0] > 1e-1:
		if i < len(opt1_asks):
			profit = profit-frac[i, 0]*asks[i]
			print("Buy {} fraction of call option on {} at strike {} with ask price {}".format(float("{0:.2f}".format(frac[i, 0])), st1, strikes[i], asks[i]))
		else: 
			profit = profit-frac[i, 0]*asks[i]
			print("Buy {} fraction of call option on {} at strike {} with ask price {}".format(float("{0:.2f}".format(frac[i, 0])), st2, strikes[i], asks[i]))
print('The maximized revenue is {}.'.format(profit))
pdb.set_trace()
#c = 


