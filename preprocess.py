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
strikes = np.linspace(strike_l, strike_h+10000, (strike_h+10000-strike_l)/500+1)
option_df = option_df[(option_df['Strike Price of the Option Times 1000']>=strike_l) & (option_df['Strike Price of the Option Times 1000']<=strike_h)]
# expiration date range
date_strike_pairs = np.array(sorted(option_df['Expiration Date of the Option'].value_counts().items()))
dates = date_strike_pairs[:, 0]
pdb.set_trace()
print('Preprocessing: using put-call parity to tighten up the call bids and asks...')
stock_price = 85400/1000
r = 0.02
for d in dates:
	# print(d)
	# plt.plot(option_df[(option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')]['Strike Price of the Option Times 1000'], \
	# 	option_df[(option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')]['Highest Closing Bid Across All Exchanges'], 'b.')
	# plt.plot(option_df[(option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')]['Strike Price of the Option Times 1000'], \
	# 	option_df[(option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')]['Lowest  Closing Ask Across All Exchanges'], 'r.')
	for k in option_df[(option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')]['Strike Price of the Option Times 1000']:
		filt = (option_df['Expiration Date of the Option']==d) & (option_df['Strike Price of the Option Times 1000']==k)
		call_bid = option_df[filt & (option_df['C=Call, P=Put']=='C')]['Highest Closing Bid Across All Exchanges'].values
		call_ask = option_df[filt & (option_df['C=Call, P=Put']=='C')]['Lowest  Closing Ask Across All Exchanges'].values
		put_bid = option_df[filt & (option_df['C=Call, P=Put']=='P')]['Highest Closing Bid Across All Exchanges'].values
		put_ask = option_df[filt & (option_df['C=Call, P=Put']=='P')]['Lowest  Closing Ask Across All Exchanges'].values
		k = k/1000
		t = d%7 + np.floor(d/7)*5
		if call_bid < stock_price-k+put_bid: 
			option_df.loc[option_df[filt & (option_df['C=Call, P=Put']=='C')].index,'Highest Closing Bid Across All Exchanges'] = \
			np.floor((stock_price-k+put_bid)*100)/100
		if call_ask > stock_price-k*np.exp(-r*t/252)+put_ask:
			option_df.loc[option_df[filt & (option_df['C=Call, P=Put']=='C')].index,'Lowest  Closing Ask Across All Exchanges'] = \
			np.ceil((stock_price-k*np.exp(-r*t/252)+put_ask)*100)/100
		# check whether bid is larger than ask
		call_bid = option_df[filt & (option_df['C=Call, P=Put']=='C')]['Highest Closing Bid Across All Exchanges'].values
		call_ask = option_df[filt & (option_df['C=Call, P=Put']=='C')]['Lowest  Closing Ask Across All Exchanges'].values
		if call_bid > call_ask:
			option_df.loc[option_df[filt & (option_df['C=Call, P=Put']=='C')].index,'Highest Closing Bid Across All Exchanges'] = np.floor(((call_bid+call_ask)/2)*100)/100
			option_df.loc[option_df[filt & (option_df['C=Call, P=Put']=='C')].index,'Lowest  Closing Ask Across All Exchanges'] = np.ceil(((call_bid+call_ask)/2)*100)/100
	# plt.plot(option_df[(option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')]['Strike Price of the Option Times 1000'], \
	# 	option_df[(option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')]['Highest Closing Bid Across All Exchanges'], 'gx')
	# plt.plot(option_df[(option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')]['Strike Price of the Option Times 1000'], \
	# 	option_df[(option_df['Expiration Date of the Option']==d) & (option_df['C=Call, P=Put']=='C')]['Lowest  Closing Ask Across All Exchanges'], 'yx')
	# pdb.set_trace()

print('Preprocessing: dropping the corresponding puts...')
option_df = option_df[option_df['C=Call, P=Put'] == 'C']

cnt = 0
count = 0
rnd = 1
while (rnd==1) | (cnt+count!=0):
	print('Round {}...'.format(rnd))
	cnt = 0
	count = 0
	print('Preprocessing: for a fixed expiration date, tightening up bids and asks across strike prices...')
	for d in dates:
		filt = option_df['Expiration Date of the Option']==d
		for i in range(0, len(option_df[filt])):
			strike = option_df[filt]['Strike Price of the Option Times 1000'].iloc[i]
			best_bid = option_df[filt]['Highest Closing Bid Across All Exchanges'].iloc[i]
			best_ask = option_df[filt]['Lowest  Closing Ask Across All Exchanges'].iloc[i]
			bid_constraint = max(option_df[filt & (option_df['Strike Price of the Option Times 1000']>=strike)]['Highest Closing Bid Across All Exchanges'])
			ask_constraint = min(option_df[filt & (option_df['Strike Price of the Option Times 1000']<=strike)]['Lowest  Closing Ask Across All Exchanges'])
			if best_bid < bid_constraint:
				# pdb.set_trace()
				option_df.loc[option_df[filt & (option_df['Strike Price of the Option Times 1000']==strike)].index, 'Highest Closing Bid Across All Exchanges'] = bid_constraint
				cnt = cnt + 1
				# pdb.set_trace()
			if best_ask > ask_constraint:
				# pdb.set_trace()
				option_df.loc[option_df[filt & (option_df['Strike Price of the Option Times 1000']==strike)].index, 'Lowest  Closing Ask Across All Exchanges'] = ask_constraint
				cnt = cnt + 1
				# pdb.set_trace()
			assert(option_df[filt]['Highest Closing Bid Across All Exchanges'].iloc[i] <= option_df[filt]['Lowest  Closing Ask Across All Exchanges'].iloc[i]), \
				"bid is larger than ask!"
	print('{} values are corrected based on bids and asks across strikes.'.format(cnt))

	print('Preprocessing: for a fixed strike price, tightening up bids and asks across expiration dates...')
	for k in strikes:
		filt = option_df['Strike Price of the Option Times 1000']==k
		for i in range(0, len(option_df[filt])):
			day = option_df[filt]['Expiration Date of the Option'].iloc[i]
			best_bid = option_df[filt]['Highest Closing Bid Across All Exchanges'].iloc[i]
			best_ask = option_df[filt]['Lowest  Closing Ask Across All Exchanges'].iloc[i]
			bid_constraint = max(option_df[filt & (option_df['Expiration Date of the Option']<=day)]['Highest Closing Bid Across All Exchanges'])
			ask_constraint = min(option_df[filt & (option_df['Expiration Date of the Option']>=day)]['Lowest  Closing Ask Across All Exchanges'])
			if best_bid < bid_constraint:
				# pdb.set_trace()
				option_df.loc[option_df[filt & (option_df['Expiration Date of the Option']==day)].index, 'Highest Closing Bid Across All Exchanges'] = bid_constraint
				count = count + 1
				# pdb.set_trace()
			if best_ask > ask_constraint:
				# pdb.set_trace()
				option_df.loc[option_df[filt & (option_df['Expiration Date of the Option']==day)].index, 'Lowest  Closing Ask Across All Exchanges'] = ask_constraint
				count = count + 1
				# pdb.set_trace()
			assert(option_df[filt]['Highest Closing Bid Across All Exchanges'].iloc[i] <= option_df[filt]['Lowest  Closing Ask Across All Exchanges'].iloc[i]), \
				"bid is larger than ask!"
	print('{} values are corrected based on bids and asks across expiration dates.'.format(count))
	rnd = rnd + 1

print('Preprocessing finished...')
date_strike_pairs = np.array(sorted(option_df['Expiration Date of the Option'].value_counts().items()))
dates = date_strike_pairs[:, 0]
dates_strikes = cumsum(date_strike_pairs[:, 1])
idx = 0
P = []
for i in range(dates_strikes.shape[0]-3):
	ymd = datetime.datetime.strptime(date, '%Y%m%d')+datetime.timedelta(days=dates[i].astype(float))
	print('Recovering the probability distribution of {} on date {}...'.format(stock, ymd.strftime("%Y-%m-%d")))
	print(option_df[['Strike Price of the Option Times 1000','Highest Closing Bid Across All Exchanges', 'Lowest  Closing Ask Across All Exchanges']].iloc[idx:dates_strikes[i]])
	c_strikes = np.concatenate([np.array(option_df['Strike Price of the Option Times 1000'].iloc[idx:dates_strikes[i]]), \
		np.array(option_df['Strike Price of the Option Times 1000'].iloc[dates_strikes[i]-1:dates_strikes[i]]+10000)])
	c_bids = np.concatenate([np.array(option_df['Highest Closing Bid Across All Exchanges'].iloc[idx:dates_strikes[i]]), [0]])
	c_asks = np.concatenate([np.array(option_df['Lowest  Closing Ask Across All Exchanges'].iloc[idx:dates_strikes[i]]), [0]])
	market_maker = constant_utility_mm.ConstantUtilityMM(c_strikes = c_strikes, c_bids = c_bids, c_asks = c_asks, stock_price = stock_price, days_to_expiration = dates[i].astype(float))
	k, p = market_maker.mm()
	expected_price = sum(k*p)
	p = np.concatenate([[np.nan]*np.where(strikes==k[0]*1000)[0][0], p, [np.nan]*(np.where(strikes==strikes[-1])[0][0] - np.where(strikes==k[-1]*1000)[0][0])])
	P.append(p)
	idx = dates_strikes[i]
	plt.plot(strikes, p, 'bx:')
	plt.title('Date {} with an expected price at {}'.format(ymd.strftime("%Y-%m-%d"), round(expected_price*100)/100))
	pdb.set_trace()

cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, dates_strikes.shape[0])]
# markers = ['o','v','d','^','D','1','2','*','3','4','8','s','p','P','h','H','+','x','X','D','|','_']
print('Plotting the probability distribution across expiration dates...')
for i in range(len(P)):
	ymd = datetime.datetime.strptime(date, '%Y%m%d')+datetime.timedelta(days=dates[i].astype(float))
	plt.plot(strikes, P[i], color=colors[i], marker = '.', linestyle=':', label=ymd.strftime("%Y-%m-%d"))
plt.title('Probability distribution of {} across expiration dates'.format(stock))
plt.legend()
pdb.set_trace()