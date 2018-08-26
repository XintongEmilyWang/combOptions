import pdb
import sys
import os
import numpy as np
import pandas as pd
import datetime, time
import math
import bisect
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
	# if (date_pairs[i,0]+date_pairs[i+1,0]) % 2 == 0:
	# 	dates.append(((date_pairs[i,0]+date_pairs[i+1,0])/2).astype(int))
dates.append(date_pairs[-1, 0])
strike_low = np.ceil(min(np.abs(opt_df['Strike Price of the Option Times 1000']))/50000)*50000
strike_high = np.ceil(max(np.abs(opt_df['Strike Price of the Option Times 1000']))/10000)*10000
strike_interval = 500

P = []
for date in dates[:-2]:
	ymd = datetime.datetime.strptime(price_date, '%Y%m%d')+datetime.timedelta(days=date.astype(float))
	print('Preprocessing: tightening the bid-ask spreads of options on {} on date {}...'.format(st, ymd.strftime("%Y-%m-%d")))
	my_spread_file = os.path.join(input_dir, st+"_spread"+"_"+ymd.strftime("%Y%m%d")+"_"+str(strike_interval)+".npy")
	if os.path.isfile(my_spread_file):
		data = np.load(my_spread_file)
		low = bisect.bisect_left(data[0], strike_low)
		high = bisect.bisect_left(data[0], strike_high)
		strikes = data[0]
		call_bids = data[1]
		call_asks = data[2]
		# coeffs = np.polyfit(strikes/1000, call_bids, 5)
		# p = np.poly1d(coeffs)
		# yp = np.polyval(p, strikes/1000)
		# plt.plot(strikes/1000, call_bids, 'bx')
		# plt.plot(strikes/1000, yp, 'r.')
		# pdb.set_trace()
		put_bids = data[3]
		put_asks = data[4]
		plt.plot(strikes/1000, call_bids, 'bx')
		plt.plot(strikes/1000, call_asks, 'bx')
		plt.plot(strikes/1000, put_bids, 'rx')
		plt.plot(strikes/1000, put_asks, 'rx')
		plt.title('{} option prices on date {}'.format(st, ymd.strftime("%Y-%m-%d")))
		# plt.show()
	else:
		spread_shrinker = single_option_bounds.SingleOptionBounds(st=st, opt_df=opt_df, strike_low=strike_low, strike_high=strike_high, strike_interval=strike_interval, days_to_expiration=date)
		strikes, call_bids, call_asks, put_bids, put_asks = spread_shrinker.tighten_spread()
		np.save(my_spread_file, [strikes, call_bids, call_asks, put_bids, put_asks])
	
	# Evaluate the tightened bid-ask spread
	opt_df_date_call = opt_df[(opt_df['Expiration Date of the Option']==date) & (opt_df['C=Call, P=Put'] == 'C') \
		& (opt_df['Strike Price of the Option Times 1000']>=strike_low) & (opt_df['Strike Price of the Option Times 1000']<=strike_high)]
	opt_df_date_put = opt_df[(opt_df['Expiration Date of the Option']==date) & (opt_df['C=Call, P=Put'] == 'P') \
		& (opt_df['Strike Price of the Option Times 1000']<=-strike_low) & (opt_df['Strike Price of the Option Times 1000']>=-strike_high)]
	total_call_spread = sum(opt_df_date_call['Lowest  Closing Ask Across All Exchanges']) - sum(opt_df_date_call['Highest Closing Bid Across All Exchanges'])
	total_put_spread = sum(opt_df_date_put['Lowest  Closing Ask Across All Exchanges']) - sum(opt_df_date_put['Highest Closing Bid Across All Exchanges'])
	call_bid_diff = np.zeros(len(opt_df_date_call))
	call_ask_diff = np.zeros(len(opt_df_date_call))
	put_bid_diff = np.zeros(len(opt_df_date_call))
	put_ask_diff = np.zeros(len(opt_df_date_call))
	for i in range(0, len(opt_df_date_call)):
		idx = bisect.bisect_left(strikes, opt_df_date_call['Strike Price of the Option Times 1000'].iloc[i])
		assert(strikes[idx]==opt_df_date_call['Strike Price of the Option Times 1000'].iloc[i]), "Strikes not equal!"
		call_bid_diff[i] = float("{0:.4f}".format(call_bids[idx] - opt_df_date_call['Highest Closing Bid Across All Exchanges'].iloc[i]))
		call_ask_diff[i] = float("{0:.4f}".format(opt_df_date_call['Lowest  Closing Ask Across All Exchanges'].iloc[i] - call_asks[idx]))
		index = bisect.bisect_left(strikes, -1*opt_df_date_put['Strike Price of the Option Times 1000'].iloc[i])
		assert(strikes[index]==-1*opt_df_date_put['Strike Price of the Option Times 1000'].iloc[i]), "Strikes not equal!"
		put_bid_diff[len(opt_df_date_call)-1-i] = float("{0:.4f}".format(put_bids[index] - opt_df_date_put['Highest Closing Bid Across All Exchanges'].iloc[i]))
		put_ask_diff[len(opt_df_date_call)-1-i] = float("{0:.4f}".format(opt_df_date_put['Lowest  Closing Ask Across All Exchanges'].iloc[i] - put_asks[index]))
	print(call_bid_diff)
	print(call_ask_diff)
	print(put_bid_diff)
	print(put_ask_diff)
	print((sum(call_bid_diff)+sum(call_ask_diff))/total_call_spread)
	print((sum(put_bid_diff)+sum(put_ask_diff))/total_put_spread)
	pdb.set_trace()

	# print('Recovering the probability distribution of {} on date {}...'.format(st, ymd.strftime("%Y-%m-%d")))
	# my_prob_file = os.path.join(input_dir, st+"_prob"+"_"+ymd.strftime("%Y%m%d")+"_"+str(strike_interval)+".npy")
	# if os.path.isfile(my_prob_file):
	# 	p = np.load(my_prob_file)
	# else:
	# 	market_maker = constant_utility_mm.ConstantUtilityMM(strikes=strikes, c_bids=call_bids, c_asks=call_asks, p_bids=put_bids, p_asks=put_asks)
	# 	p = market_maker.mm()
	# 	np.save(my_prob_file, p)
	# expected_price = sum(strikes*p)/1000
	# P.append(p)
	# plt.plot(strikes[:-10]/1000, p[:-10], 'bx:')
	# plt.title('{} on date {} with an expected price at {}'.format(st, ymd.strftime("%Y-%m-%d"), round(expected_price*100)/100))
	# plt.show()
pdb.set_trace()
cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, len(dates))]
# markers = ['o','v','d','^','D','1','2','*','3','4','8','s','p','P','h','H','+','x','X','D','|','_']
print('Plotting the probability distribution across expiration dates...')
for i in range(len(P)):
	ymd = datetime.datetime.strptime(price_date, '%Y%m%d')+datetime.timedelta(days=dates[i].astype(float))
	plt.plot(strikes[:-10]/1000, P[i][:-10], color=colors[i], marker = '.', linestyle=':', label=ymd.strftime("%Y-%m-%d"))
plt.title('Probability distribution of {} across expiration dates'.format(st))
plt.xlabel('Stock Price')
plt.ylabel('Probability')
plt.legend()
pdb.set_trace()