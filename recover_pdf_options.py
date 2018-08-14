import pdb
import sys
import os
import numpy as np
import pandas as pd
import datetime, time
import math
import single_option_bounds
import constant_utility_mm
import matplotlib.pyplot as plt
from gurobipy import *

# Given the current price of Option(ST, K, T) in the market
# Return the pricing bound of Option(ST) on any strike k and any expiration date t
input_dir = sys.argv[1]
st = sys.argv[2]
price_date = sys.argv[3]

opt_stats_filename = os.path.join(input_dir, st+"_"+price_date+".xlsx")
opt_df = pd.read_excel(opt_stats_filename)
assert(np.array(opt_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day."

# Preprocessing - convert puts to calls
print('Preprocessing: converting puts P(S,K,T) to the corresponding calls C(-S, -K, T)...')
unit_col = np.array(opt_df['C=Call, P=Put'] == 'C').astype(int) - np.array(opt_df['C=Call, P=Put'] == 'P').astype(int)
opt_df = opt_df.assign(Unit=unit_col)
opt_df['Strike Price of the Option Times 1000'] = opt_df['Strike Price of the Option Times 1000'].astype(int)
opt_df.loc[opt_df['C=Call, P=Put'] == 'P', 'Strike Price of the Option Times 1000'] = -1 * opt_df[opt_df['C=Call, P=Put'] == 'P']['Strike Price of the Option Times 1000']
opt_df['Expiration Date of the Option'] = ((pd.to_datetime(opt_df['Expiration Date of the Option'], format = '%Y%m%d') - \
datetime.datetime.strptime(price_date, '%Y%m%d')) / np.timedelta64(1, 'D')).astype(int)
opt_df['Highest Closing Bid Across All Exchanges'] = opt_df['Highest Closing Bid Across All Exchanges'].astype(float)
opt_df['Lowest  Closing Ask Across All Exchanges'] = opt_df['Lowest  Closing Ask Across All Exchanges'].astype(float)
opt_df = opt_df[['Expiration Date of the Option', 'C=Call, P=Put', 'Unit', 'Strike Price of the Option Times 1000', \
	'Highest Closing Bid Across All Exchanges', 'Lowest  Closing Ask Across All Exchanges']]
opt_df.sort_values(by = ['Expiration Date of the Option', 'C=Call, P=Put', 'Strike Price of the Option Times 1000'], inplace = True)

# Get the strike and date range interested
print('Preprocessing: getting the range of expiration dates and strikes...')
date_pairs = np.array(sorted(opt_df['Expiration Date of the Option'].value_counts().items()))
dates = []
for i in range(0, len(date_pairs[:, 0])-1):
	dates.append(date_pairs[i,0])
	if (date_pairs[i,0]+date_pairs[i+1,0]) % 2 == 0:
		dates.append(((date_pairs[i,0]+date_pairs[i+1,0])/2).astype(int))
dates.append(date_pairs[-1, 0])
strike_low = np.ceil(min(np.abs(opt_df['Strike Price of the Option Times 1000']))/50000)*50000
strike_high = np.ceil(max(np.abs(opt_df['Strike Price of the Option Times 1000']))/10000)*10000
strike_interval = 500

P = []
for date in dates[8:]:
	ymd = datetime.datetime.strptime(price_date, '%Y%m%d')+datetime.timedelta(days=date.astype(float))
	print('Preprocessing: tightening the bid-ask spreads of options on {} on date {}...'.format(st, ymd.strftime("%Y-%m-%d")))
	my_spread_file = os.path.join(input_dir, st+"_spread"+"_"+ymd.strftime("%Y%m%d")+"_"+str(strike_interval)+".npy")
	if os.path.isfile(my_spread_file):
		data = np.load(my_spread_file)
		strikes = data[0]
		call_bids = data[1]
		call_asks = data[2]
		put_bids = data[3]
		put_asks = data[4]
	else:
		spread_shrinker = single_option_bounds.SingleOptionBounds(st=st, opt_df=opt_df, strike_low=strike_low, strike_high=strike_high, strike_interval=strike_interval, days_to_expiration=date)
		strikes, call_bids, call_asks, put_bids, put_asks = spread_shrinker.tighten_spread()
		np.save(my_spread_file, [strikes, call_bids, call_asks, put_bids, put_asks])
	
	print('Recovering the probability distribution of {} on date {}...'.format(st, ymd.strftime("%Y-%m-%d")))
	my_prob_file = os.path.join(input_dir, st+"_prob"+"_"+ymd.strftime("%Y%m%d")+"_"+str(strike_interval)+".npy")
	if os.path.isfile(my_prob_file):
		p = np.load(my_prob_file)
	else:
		market_maker = constant_utility_mm.ConstantUtilityMM(strikes=strikes, c_bids=call_bids, c_asks=call_asks, p_bids=put_bids, p_asks=put_asks)
		p = market_maker.mm()
		np.save(my_prob_file, p)
	expected_price = sum(strikes*p)/1000
	P.append(p)
	# plt.plot(strikes/1000, p, 'bx:')
	# plt.title('{} on date {} with an expected price at {}'.format(st, ymd.strftime("%Y-%m-%d"), round(expected_price*100)/100))
pdb.set_trace()
cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, dates.shape[0])]
# markers = ['o','v','d','^','D','1','2','*','3','4','8','s','p','P','h','H','+','x','X','D','|','_']
print('Plotting the probability distribution across expiration dates...')
for i in range(len(P)):
	ymd = datetime.datetime.strptime(date, '%Y%m%d')+datetime.timedelta(days=dates[i].astype(float))
	plt.plot(strikes, P[i], color=colors[i], marker = '.', linestyle=':', label=ymd.strftime("%Y-%m-%d"))
plt.title('Probability distribution of {} across expiration dates'.format(stock))
plt.legend()
pdb.set_trace()