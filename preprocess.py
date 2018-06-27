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
# change date to the number of days to expiration
option_df['Expiration Date of the Option'] = ((pd.to_datetime(option_df['Expiration Date of the Option'], format = '%Y%m%d') - \
datetime.datetime.strptime(date, '%Y%m%d')) / np.timedelta64(1, 'D')).astype(int)
assert(np.array(option_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"
option_df.sort_values(by = ['Expiration Date of the Option', 'C=Call, P=Put', \
	'Strike Price of the Option Times 1000'], inplace = True)
# trim the strike range
strike_l, h = stats.t.interval(0.68, len(option_df['Strike Price of the Option Times 1000'])-1, loc=np.mean(option_df['Strike Price of the Option Times 1000']), \
	scale=np.std(option_df['Strike Price of the Option Times 1000']))
strike_l = np.ceil(strike_l/500)*500
l, strike_h = stats.t.interval(0.95, len(option_df['Strike Price of the Option Times 1000'])-1, loc=np.mean(option_df['Strike Price of the Option Times 1000']), \
	scale=np.std(option_df['Strike Price of the Option Times 1000']))
strike_h = np.floor(strike_h/500)*500
strikes = np.linspace(strike_l, strike_h, (strike_h-strike_l)/500+1)
option_df = option_df[(option_df['Strike Price of the Option Times 1000']>=strike_l) & (option_df['Strike Price of the Option Times 1000']<=strike_h)]

# expiration date range
date_strike_pairs = np.array(sorted(option_df['Expiration Date of the Option'].value_counts().items()))
dates = date_strike_pairs[:, 0]

# using the put-call parity to tighten up the call bids and asks
stock_price = 85400/1000
r = 0.02
for d in dates:
	# print(d)
	# pdb.set_trace()
	for k in option_df[(option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')]['Strike Price of the Option Times 1000']:
		# print(k)
		filt = (option_df['Expiration Date of the Option']==d) & (option_df['Strike Price of the Option Times 1000']==k)
		call_bid = option_df[filt & (option_df['C=Call, P=Put']=='C')]['Highest Closing Bid Across All Exchanges'].values
		call_ask = option_df[filt & (option_df['C=Call, P=Put']=='C')]['Lowest  Closing Ask Across All Exchanges'].values
		put_bid = option_df[filt & (option_df['C=Call, P=Put']=='P')]['Highest Closing Bid Across All Exchanges'].values
		put_ask = option_df[filt & (option_df['C=Call, P=Put']=='P')]['Lowest  Closing Ask Across All Exchanges'].values
		k = k/1000
		t = d%7 + np.floor(d/7)*5
		if call_bid < stock_price-k+put_bid: 
			# pdb.set_trace()
			option_df.loc[option_df[filt & (option_df['C=Call, P=Put']=='C')].index,'Highest Closing Bid Across All Exchanges'] = \
			np.floor((stock_price-k+put_bid)*100)/100
			# pdb.set_trace()
		if call_ask > stock_price-k*np.exp(-r*t/252)+put_ask:
			# pdb.set_trace()
			option_df.loc[option_df[filt & (option_df['C=Call, P=Put']=='C')].index,'Lowest  Closing Ask Across All Exchanges'] = \
			np.ceil((stock_price-k*np.exp(-r*t/252)+put_ask)*100)/100
			# pdb.set_trace()
		# check whether bid is larger than ask
		call_bid = option_df[filt & (option_df['C=Call, P=Put']=='C')]['Highest Closing Bid Across All Exchanges'].values
		call_ask = option_df[filt & (option_df['C=Call, P=Put']=='C')]['Lowest  Closing Ask Across All Exchanges'].values
		if call_bid > call_ask:
			# pdb.set_trace()
			option_df.loc[option_df[filt & (option_df['C=Call, P=Put']=='C')].index,'Highest Closing Bid Across All Exchanges'] = np.floor(((call_bid+call_ask)/2)*100)/100
			option_df.loc[option_df[filt & (option_df['C=Call, P=Put']=='C')].index,'Lowest  Closing Ask Across All Exchanges'] = np.ceil(((call_bid+call_ask)/2)*100)/100
			# pdb.set_trace()
	# pdb.set_trace()

# drop all the puts
option_df = option_df[option_df['C=Call, P=Put'] == 'C']

# For a fixed expiration date, tighten up bids and asks across strike prices
for d in dates:
	filt = option_df['Expiration Date of the Option']==d
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

# For a fixed strike price, tighten up bids and asks across expiration dates
for k in strikes:
	filt = option_df['Strike Price of the Option Times 1000']==k
	for i in range(0, len(option_df[filt])):
		day = option_df[filt]['Expiration Date of the Option'].iloc[i]
		best_bid = option_df[filt]['Highest Closing Bid Across All Exchanges'].iloc[i]
		best_ask = option_df[filt]['Lowest  Closing Ask Across All Exchanges'].iloc[i]
		bid_constraint = max(option_df[filt & (option_df['Expiration Date of the Option']<=day)]['Highest Closing Bid Across All Exchanges'])
		ask_constraint = min(option_df[filt & (option_df['Expiration Date of the Option']>=day)]['Lowest  Closing Ask Across All Exchanges'])
		if best_bid < bid_constraint:
			# push the best bid up
			#pdb.set_trace()
			option_df.loc[option_df[filt & (option_df['Expiration Date of the Option']==day)].index, 'Highest Closing Bid Across All Exchanges'] = bid_constraint
			#pdb.set_trace()
		if best_ask > ask_constraint:
			# push the best ask down
			#pdb.set_trace()
			option_df.loc[option_df[filt & (option_df['Expiration Date of the Option']==day)].index, 'Lowest  Closing Ask Across All Exchanges'] = ask_constraint
			#pdb.set_trace()

date_strike_pairs = np.array(sorted(option_df['Expiration Date of the Option'].value_counts().items()))
dates = date_strike_pairs[:, 0]
dates_strikes = cumsum(date_strike_pairs[:, 1])

idx = 0
P = []
for i in range(dates_strikes.shape[0]):
	print('Recovering the probability distribution of {} on date {}'.format(stock, dates[i]))
	print(option_df[['Strike Price of the Option Times 1000','Highest Closing Bid Across All Exchanges', 'Lowest  Closing Ask Across All Exchanges']].iloc[idx:dates_strikes[i]])
	c_strikes = np.array(option_df['Strike Price of the Option Times 1000'].iloc[idx:dates_strikes[i]])
	c_bids = np.array(option_df['Highest Closing Bid Across All Exchanges'].iloc[idx:dates_strikes[i]])
	c_asks = np.array(option_df['Lowest  Closing Ask Across All Exchanges'].iloc[idx:dates_strikes[i]])
	market_maker = constant_utility_mm.ConstantUtilityMM(c_strikes = c_strikes, c_bids = c_bids, c_asks = c_asks)
	k, p = market_maker.mm()
	expected_price = sum(k*p)
	p = np.concatenate([[np.nan]*np.where(strikes==k[0]*1000)[0][0], p, [np.nan]*(np.where(strikes==strikes[-1])[0][0] - np.where(strikes==k[-1]*1000)[0][0])])
	P.append(p)
	idx = dates_strikes[i]
	plt.plot(strikes, p, 'bx')
	ymd = datetime.datetime.strptime(date, '%Y%m%d')+datetime.timedelta(days=dates[i].astype(float))
	plt.title('Date {} with an expected price at {}'.format(ymd.strftime("%Y-%m-%d"), round(expected_price*100)/100))
	pdb.set_trace()

# plot the recovered probability distribution across time
pdb.set_trace()