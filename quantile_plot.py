import pdb
import sys
import os
import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import datetime, time

input_dir = sys.argv[1]
stock = sys.argv[2]
date = sys.argv[3]

prob = np.load('prob.npy')
prob[isnan(prob)] = 0
prob = np.round_(prob*1e6).astype(int)
strikes = np.load('strikes.npy')
dates = np.load('dates.npy')
freqs = []
quantiles = np.linspace(10, 90, (90-10)/10+1).astype(int)
for i in range(0, len(prob)):
	freqs.append(np.percentile(np.repeat(strikes, prob[i]), quantiles))
freqs = np.array(freqs)

stock_hist_filename = os.path.join(input_dir, stock+"_"+"hist.xlsx")
stock_hist_df = pd.read_excel(stock_hist_filename)
# change date to the number of days to expiration
stock_hist_df['The Date for this Price Record'] = ((pd.to_datetime(stock_hist_df['The Date for this Price Record'], format = '%Y%m%d') - \
datetime.datetime.strptime(date, '%Y%m%d')) / np.timedelta64(1, 'D')).astype(int)
hist_dates = np.concatenate([[0], -1*dates[:-3]-1])
hist_prices = []
for d in hist_dates:
	hist_prices.append(int(stock_hist_df[stock_hist_df['The Date for this Price Record'] == d]['Close (or Bid-Ask Average if Negative)']*1000))

cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, quantiles.shape[0])]
plt.plot(hist_dates, hist_prices, color = 'black', marker = '.', linestyle=':', label = 'past prices')
for i in range(0, quantiles.shape[0]):
	plt.plot(dates[:-3], freqs[:, i], color=colors[i], marker = '.', linestyle=':', label = '{}th quantile'.format(quantiles[i]))

plt.legend()
plt.title('{} past and projected prices at {} across time.'.format(stock, date))
pdb.set_trace()