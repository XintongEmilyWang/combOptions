import pdb
import sys
import os
import numpy as np
from numpy import cumsum
import pandas as pd
import scipy.stats as stats
import constant_utility_mm
import matplotlib.pyplot as plt
import datetime, time

# three components: option, underlying stock, dividend
input_dir = sys.argv[1]
stock = sys.argv[2]
date = sys.argv[3]

option_stats_filename = os.path.join(input_dir, stock+"_"+date+".xlsx")
option_df = pd.read_excel(option_stats_filename)
option_df['Strike Price of the Option Times 1000'] = option_df['Strike Price of the Option Times 1000'].astype(int)
option_df['Highest Closing Bid Across All Exchanges'] = option_df['Highest Closing Bid Across All Exchanges'].astype(float)
option_df['Lowest  Closing Ask Across All Exchanges'] = option_df['Lowest  Closing Ask Across All Exchanges'].astype(float)
option_df['Expiration Date of the Option'] = ((pd.to_datetime(option_df['Expiration Date of the Option'], format = '%Y%m%d') - \
datetime.datetime.strptime(date, '%Y%m%d')) / np.timedelta64(1, 'D')).astype(int)
assert(np.array(option_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"
option_df.sort_values(by = ['Expiration Date of the Option', 'C=Call, P=Put', \
	'Strike Price of the Option Times 1000'], inplace = True)

# For a fixed expiration date, tighten up bids and asks across strike prices
date_strike_pairs = np.array(sorted(option_df['Expiration Date of the Option'].value_counts().items()))
dates = date_strike_pairs[:, 0]
for d in dates:
	filt = (option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')
	for i in range(0, len(option_df[filt])):
		strike = option_df[filt]['Strike Price of the Option Times 1000'].iloc[i]
		best_bid = option_df[filt]['Highest Closing Bid Across All Exchanges'].iloc[i]
		best_ask = option_df[filt]['Lowest  Closing Ask Across All Exchanges'].iloc[i]
		bid_constraint = max(option_df[filt & (option_df['Strike Price of the Option Times 1000']>=strike)]['Highest Closing Bid Across All Exchanges'])
		ask_constraint = min(option_df[filt & (option_df['Strike Price of the Option Times 1000']<=strike)]['Lowest  Closing Ask Across All Exchanges'])
		if best_bid < bid_constraint:
			# push the best bid up
			#pdb.set_trace()
			option_df.loc[option_df[filt & (option_df['Strike Price of the Option Times 1000']==strike)].index, 'Highest Closing Bid Across All Exchanges'] = bid_constraint
			#pdb.set_trace()
		if best_ask > ask_constraint:
			# push the best ask down
			#pdb.set_trace()
			option_df.loc[option_df[filt & (option_df['Strike Price of the Option Times 1000']==strike)].index, 'Lowest  Closing Ask Across All Exchanges'] = ask_constraint
			#pdb.set_trace()
	#pdb.set_trace()

# For a fixed strike price, 
strike_date_pairs = np.array(sorted(option_df['Strike Price of the Option Times 1000'].value_counts().items()))
strikes_l, strike_h = st.t.interval(0.95, len(option_df['Strike Price of the Option Times 1000'])-1, loc=np.mean(option_df['Strike Price of the Option Times 1000']), \
	scale=np.std(option_df['Strike Price of the Option Times 1000']))
pdb.set_trace()
option_subset_df = option_df[option_df['Strike Price of the Option Times 1000'] == 70000]
option_time_df = option_subset_df[option_df['C=Call, P=Put'] == 'C'][['Expiration Date of the Option', \
'Highest Closing Bid Across All Exchanges', 'Lowest  Closing Ask Across All Exchanges']]
pdb.set_trace()
date_strike_pairs = np.array(sorted(option_df['Expiration Date of the Option'].value_counts().items()))
dates = date_strike_pairs[:, 0]
dates_strikes = cumsum(date_strike_pairs[:, 1])
idx = 0
K = []
P = []
for i in range(dates_strikes.shape[0]):
	print('Recovering the probability distribution of {} on date {}'.format(stock, dates[i]))
	c_p = np.array(option_df['C=Call, P=Put'].iloc[idx:dates_strikes[i]].value_counts())
	c_strikes = np.array(option_df['Strike Price of the Option Times 1000'].iloc[idx:idx+c_p[0]])
	c_bids = np.array(option_df['Highest Closing Bid Across All Exchanges'].iloc[idx:idx+c_p[0]])
	c_asks = np.array(option_df['Lowest  Closing Ask Across All Exchanges'].iloc[idx:idx+c_p[0]])
	p_strikes = np.array(option_df['Strike Price of the Option Times 1000'].iloc[idx+c_p[0]:dates_strikes[i]])
	p_bids = np.array(option_df['Highest Closing Bid Across All Exchanges'].iloc[idx+c_p[0]:dates_strikes[i]])
	p_asks = np.array(option_df['Lowest  Closing Ask Across All Exchanges'].iloc[idx+c_p[0]:dates_strikes[i]])

	market_maker = constant_utility_mm.ConstantUtilityMM(c_strikes = c_strikes, \
		c_bids = c_bids, c_asks = c_asks, p_strikes = p_strikes, p_bids = p_bids, p_asks = p_asks)
	k, p = market_maker.mm()
	K.append(k)
	P.append(p)
	idx = dates_strikes[i]
	pdb.set_trace()

pdb.set_trace()