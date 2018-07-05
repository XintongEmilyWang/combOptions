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
for i in range(0, len(prob)):
	freqs.append(np.percentile(np.repeat(strikes, prob[i]), [10, 20, 30, 40, 50, 60, 70, 80, 90]))
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

plt.plot(hist_dates, hist_prices, color = 'black', marker = '.', linestyle=':', label = 'past prices')
plt.plot(dates[:-3], freqs[:, 0], 'b.', linestyle=':', label = '10 quantile')
plt.plot(dates[:-3], freqs[:, 1], 'r.', linestyle=':', label = '20 quantile')
plt.plot(dates[:-3], freqs[:, 2], 'g.', linestyle=':', label = '30 quantile')
plt.plot(dates[:-3], freqs[:, 3], 'b.', linestyle=':', label = '40 quantile')
plt.plot(dates[:-3], freqs[:, 4], 'r.', linestyle=':', label = '50 quantile')
plt.plot(dates[:-3], freqs[:, 5], 'g.', linestyle=':', label = '60 quantile')
plt.plot(dates[:-3], freqs[:, 6], 'b.', linestyle=':', label = '70 quantile')
plt.plot(dates[:-3], freqs[:, 7], 'r.', linestyle=':', label = '80 quantile')
plt.plot(dates[:-3], freqs[:, 8], 'g.', linestyle=':', label = '90 quantile')
plt.legend()
plt.title('{} past and projected prices at {} across time.'.format(stock, date))
pdb.set_trace()