import pdb
import sys
import os
import numpy as np
from numpy import cumsum
import pandas as pd
import constant_utility_mm

# three components: option, underlying stock, dividend
input_dir = sys.argv[1]
stock = sys.argv[2]
date = sys.argv[3]

option_stats_filename = os.path.join(input_dir, stock+"_"+date+".xlsx")
option_df = pd.read_excel(option_stats_filename)
option_df['Strike Price of the Option Times 1000'] = option_df['Strike Price of the Option Times 1000'].astype(int)
option_df['Highest Closing Bid Across All Exchanges'] = option_df['Highest Closing Bid Across All Exchanges'].astype(float)
option_df['Lowest  Closing Ask Across All Exchanges'] = option_df['Lowest  Closing Ask Across All Exchanges'].astype(float)
assert(np.array(option_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"
option_df.sort_values(by = ['Expiration Date of the Option', 'C=Call, P=Put', \
	'Strike Price of the Option Times 1000'], inplace = True)
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