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
import math

# three components: option, underlying stock, dividend
input_dir = sys.argv[1]
price_date = sys.argv[2]
st1 = sys.argv[3]
st2 = sys.argv[4]
expiration_date = int(sys.argv[5])
comb_type = sys.argv[6]

opt1_stats_filename = os.path.join(input_dir, st1+"_"+price_date+".xlsx")
opt1_df = pd.read_excel(opt1_stats_filename)
opt2_stats_filename = os.path.join(input_dir, st2+"_"+price_date+".xlsx")
opt2_df = pd.read_excel(opt2_stats_filename)

assert(np.array(opt1_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"
assert(np.array(opt2_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"

opt1_df = opt1_df[opt1_df['Expiration Date of the Option']==expiration_date]
opt2_df = opt2_df[opt2_df['Expiration Date of the Option']==expiration_date]
opt1_df['Strike Price of the Option Times 1000'] = opt1_df['Strike Price of the Option Times 1000'].astype(int)
opt1_df['Highest Closing Bid Across All Exchanges'] = opt1_df['Highest Closing Bid Across All Exchanges'].astype(float)
opt1_df['Lowest  Closing Ask Across All Exchanges'] = opt1_df['Lowest  Closing Ask Across All Exchanges'].astype(float)
opt2_df['Strike Price of the Option Times 1000'] = opt2_df['Strike Price of the Option Times 1000'].astype(int)
opt2_df['Highest Closing Bid Across All Exchanges'] = opt2_df['Highest Closing Bid Across All Exchanges'].astype(float)
opt2_df['Lowest  Closing Ask Across All Exchanges'] = opt2_df['Lowest  Closing Ask Across All Exchanges'].astype(float)
opt1_df.sort_values(by = ['C=Call, P=Put', 'Strike Price of the Option Times 1000'], inplace = True)
opt2_df.sort_values(by = ['C=Call, P=Put', 'Strike Price of the Option Times 1000'], inplace = True)

addition_l = min(opt1_df['Strike Price of the Option Times 1000']) + min(opt2_df['Strike Price of the Option Times 1000'])
addition_h = max(opt1_df['Strike Price of the Option Times 1000']) + max(opt2_df['Strike Price of the Option Times 1000'])
subtraction_l = min(opt1_df['Strike Price of the Option Times 1000']) - max(opt2_df['Strike Price of the Option Times 1000'])
subtraction_h = max(opt1_df['Strike Price of the Option Times 1000']) - min(opt2_df['Strike Price of the Option Times 1000'])

if comb_type == 'C+':
	ub = []
	lb = []
	print('Finding constraints for C({}+{}) on date {}...'.format(st1, st2, expiration_date))
	for comb_strike in np.linspace(addition_h, addition_l, (addition_h-addition_l)/5000+1).astype(int):
		upper = math.inf
		lower = 0
		upper_k = math.inf
		lower_k = math.inf
		lower_type = ''

		opt1_df = opt1_df[opt1_df['Strike Price of the Option Times 1000'] <= comb_strike]
		opt2_df = opt2_df[opt2_df['Strike Price of the Option Times 1000'] <= comb_strike]

		# check each single call option with the comb_strike
		f1 = (opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == comb_strike)
		if any(f1):
			lower = max(lower, float(opt1_df[f1]['Highest Closing Bid Across All Exchanges']))
		f2 = (opt1_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike)
		if any(f2):
			lower = max(lower, float(opt2_df[f2]['Highest Closing Bid Across All Exchanges']))

		for k in opt1_df[opt1_df['C=Call, P=Put'] == 'C']['Strike Price of the Option Times 1000']:
			filt1c = (opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == k)
			filt1p = (opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == k)
			filt2c = (opt2_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-k)
			filt2p = (opt2_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-k)
			if any(filt2c):
				upper_temp = float(opt1_df[filt1c]['Lowest  Closing Ask Across All Exchanges']) + float(opt2_df[filt2c]['Lowest  Closing Ask Across All Exchanges'])
				if upper_temp < upper:
					upper_k = k
					upper = upper_temp
					upper = float("{0:.2f}".format(upper))
				lower_temp = float(opt2_df[filt2c]['Highest Closing Bid Across All Exchanges']) - float(opt1_df[filt1p]['Lowest  Closing Ask Across All Exchanges'])
				if lower_temp > lower:
					lower_k = k
					lower = float("{0:.2f}".format(lower_temp))
					lower_type = 'P'
			if any(filt2p):
				lower_temp = float(opt1_df[filt1c]['Highest Closing Bid Across All Exchanges']) - float(opt2_df[filt2p]['Lowest  Closing Ask Across All Exchanges'])
				if lower_temp > lower:
					lower_k = k
					lower = float("{0:.2f}".format(lower_temp))
					lower_type = 'C'
		ub.append(upper)
		lb.append(lower)
		print('{} <= C({}+{}, {}) <= {}'.format(lower, st1, st2, comb_strike, upper))
		if lower_type == '':
			print('Lower bound 0: by definition of an option.')
		elif lower_type == 'C':
			print('Lower bound {}: C({}, {}) = {} - P({}, {}) = {}'.format(lower, st1, lower_k, \
				float(opt1_df[(opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == lower_k)]['Highest Closing Bid Across All Exchanges']), \
				st2, comb_strike-lower_k, \
				float(opt2_df[(opt2_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-lower_k)]['Lowest  Closing Ask Across All Exchanges'])))
		else:
			print('Lower bound {}: C({}, {}) = {} - P({}, {}) = {}'.format(lower, st2, comb_strike-lower_k, \
				float(opt2_df[(opt2_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-lower_k)]['Highest Closing Bid Across All Exchanges']), \
				st1, lower_k, \
				float(opt1_df[(opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == lower_k)]['Lowest  Closing Ask Across All Exchanges'])))
		
		print('Upper bound {}: C({}, {}) = {} + C({}, {}) = {}'.format(upper, st1, upper_k, \
			float(opt1_df[(opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == upper_k)]['Lowest  Closing Ask Across All Exchanges']), \
			st2, comb_strike-upper_k, \
			float(opt2_df[(opt2_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-upper_k)]['Lowest  Closing Ask Across All Exchanges'])))
		print('')
	
	plt.plot(np.linspace(addition_h, addition_l, (addition_h-addition_l)/5000+1).astype(int)/1000, ub, 'r.')
	plt.plot(np.linspace(addition_h, addition_l, (addition_h-addition_l)/5000+1).astype(int)/1000, lb, 'b.')
	plt.title('Price bounds on call options of {} + {} across strikes.'.format(st1, st2))
	plt.show()

if comb_type == 'C-':
	ub = []
	lb = []
	print('Finding constraints for C({}-{}) on date {}...'.format(st1, st2, expiration_date))
	for comb_strike in np.linspace(subtraction_l, subtraction_h, (subtraction_h-subtraction_l)/5000+1).astype(int):
		upper = math.inf
		lower = 0
		upper_k = math.inf
		lower_k = math.inf
		lower_type = ''

		standard = (opt1_df['Strike Price of the Option Times 1000'] >= min(opt2_df['Strike Price of the Option Times 1000']) + comb_strike) & \
			(opt1_df['Strike Price of the Option Times 1000'] <= max(opt2_df['Strike Price of the Option Times 1000']) + comb_strike)
		for k in opt1_df[standard & (opt1_df['C=Call, P=Put'] == 'C')]['Strike Price of the Option Times 1000']:
			filt1c = (opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == k)
			filt1p = (opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == k)
			filt2c = (opt2_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == k-comb_strike)
			filt2p = (opt2_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == k-comb_strike)
			if any(filt2c):
				lower_temp = float(opt1_df[filt1c]['Highest Closing Bid Across All Exchanges']) - float(opt2_df[filt2c]['Lowest  Closing Ask Across All Exchanges'])
				if lower_temp > lower:
					lower = float("{0:.2f}".format(lower_temp))
					lower_k = k
					lower_type = 'C'
			if any(filt2p):
				upper_temp = float(opt1_df[filt1c]['Lowest  Closing Ask Across All Exchanges']) + float(opt2_df[filt2p]['Lowest  Closing Ask Across All Exchanges'])
				if upper_temp < upper:
					upper = float("{0:.2f}".format(upper_temp))
					upper_k = k
				lower_temp = float(opt2_df[filt2p]['Highest Closing Bid Across All Exchanges']) - float(opt1_df[filt1p]['Lowest  Closing Ask Across All Exchanges'])
				if lower_temp > lower:
					lower = float("{0:.2f}".format(lower_temp))
					lower_k = k
					lower_type = 'P'
		ub.append(upper)
		lb.append(lower)
		print('{} <= C({}-{}, {}) <= {}'.format(lower, st1, st2, comb_strike, upper))
		if lower_type == '':
			print('Lower bound 0: by definition of an option.')
		elif lower_type == 'C':
			print('Lower bound {}: C({}, {}) = {} - C({}, {}) = {}'.format(lower, st1, lower_k, \
				float(opt1_df[(opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == lower_k)]['Highest Closing Bid Across All Exchanges']), \
				st2, lower_k-comb_strike, \
				float(opt2_df[(opt2_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == lower_k-comb_strike)]['Lowest  Closing Ask Across All Exchanges'])))
		else:
			print('Lower bound {}: P({}, {}) = {} - P({}, {}) = {}'.format(lower, st2, lower_k-comb_strike, \
				float(opt2_df[(opt2_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-lower_k)]['Highest Closing Bid Across All Exchanges']), \
				st1, lower_k, \
				float(opt1_df[(opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == lower_k)]['Lowest  Closing Ask Across All Exchanges'])))
		
		print('Upper bound {}: C({}, {}) = {} + P({}, {}) = {}'.format(upper, st1, upper_k, \
			float(opt1_df[(opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == upper_k)]['Lowest  Closing Ask Across All Exchanges']), \
			st2, upper_k-comb_strike, \
			float(opt2_df[(opt2_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == upper_k-comb_strike)]['Lowest  Closing Ask Across All Exchanges'])))
		print('')

	plt.plot(np.linspace(subtraction_l, subtraction_h, (subtraction_h-subtraction_l)/5000+1).astype(int)/1000, ub, 'r.')
	plt.plot(np.linspace(subtraction_l, subtraction_h, (subtraction_h-subtraction_l)/5000+1).astype(int)/1000, lb, 'b.')
	plt.title('Price bounds on call options of {} - {} across strikes.'.format(st1, st2))
	plt.show()

if comb_type == 'P+':
	ub = []
	lb = []
	print('Finding constraints for P({}+{}) on date {}...'.format(st1, st2, expiration_date))
	for comb_strike in np.linspace(addition_h, addition_l, (addition_h-addition_l)/5000+1).astype(int):
		upper = math.inf
		lower = 0
		upper_k = math.inf
		lower_k = math.inf
		lower_type = ''
		
		opt1_df = opt1_df[opt1_df['Strike Price of the Option Times 1000'] <= comb_strike]
		opt2_df = opt2_df[opt2_df['Strike Price of the Option Times 1000'] <= comb_strike]
		# check each single put option with the comb_strike
		f1 = (opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == comb_strike)
		if any(f1):
			upper = min(upper, float(opt1_df[f1]['Lowest  Closing Ask Across All Exchanges']))
		f2 = (opt1_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike)
		if any(f2):
			upper = min(upper, float(opt2_df[f2]['Lowest  Closing Ask Across All Exchanges']))

		for k in opt1_df[opt1_df['C=Call, P=Put'] == 'P']['Strike Price of the Option Times 1000']:
			filt1c = (opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == k)
			filt1p = (opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == k)
			filt2c = (opt2_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-k)
			filt2p = (opt2_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-k)
			if any(filt2c):
				lower_temp = float(opt1_df[filt1p]['Highest Closing Bid Across All Exchanges']) - float(opt2_df[filt2c]['Lowest  Closing Ask Across All Exchanges'])
				if lower_temp > lower:
					lower = float("{0:.2f}".format(lower_temp))
					lower_k = k
					lower_type = 'P'
			if any(filt2p):
				upper_temp = float(opt1_df[filt1p]['Lowest  Closing Ask Across All Exchanges']) + float(opt2_df[filt2p]['Lowest  Closing Ask Across All Exchanges'])
				if upper_temp < upper:
					upper = float("{0:.2f}".format(upper_temp))
					upper_k = k
				lower_temp = float(opt2_df[filt2p]['Highest Closing Bid Across All Exchanges']) - float(opt1_df[filt1c]['Lowest  Closing Ask Across All Exchanges'])
				if lower_temp > lower:
					lower = float("{0:.2f}".format(lower_temp))
					lower_k = k
					lower_type = 'C'
		ub.append(upper)
		lb.append(lower)
		print('{} <= P({}+{}, {}) <= {}'.format(lower, st1, st2, comb_strike, upper))
		if lower_type == '':
			print('Lower bound 0: by definition of an option.')
		elif lower_type == 'C':
			print('Lower bound {}: P({}, {}) = {} - C({}, {}) = {}'.format(lower, st2, comb_strike-lower_k, \
				float(opt2_df[(opt2_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-lower_k)]['Highest Closing Bid Across All Exchanges']), \
				st1, lower_k, \
				float(opt1_df[(opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == lower_k)]['Lowest  Closing Ask Across All Exchanges'])))
		else:
			print('Lower bound {}: P({}, {}) = {} - C({}, {}) = {}'.format(lower, st1, lower_k, \
				float(opt1_df[(opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == lower_k)]['Highest Closing Bid Across All Exchanges']), \
				st2, comb_strike-lower_k, \
				float(opt2_df[(opt2_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-lower_k)]['Lowest  Closing Ask Across All Exchanges'])))
		#TO-DO
		if upper_k == math.inf:
			print('Upper bound {}: put from a single stock'.format(upper))
		else:
			print('Upper bound {}: P({}, {}) = {} + P({}, {}) = {}'.format(upper, st1, upper_k, \
				float(opt1_df[(opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == upper_k)]['Lowest  Closing Ask Across All Exchanges']), \
				st2, comb_strike-upper_k, \
				float(opt2_df[(opt2_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == comb_strike-upper_k)]['Lowest  Closing Ask Across All Exchanges'])))
		print('')

	plt.plot(np.linspace(addition_h, addition_l, (addition_h-addition_l)/5000+1).astype(int)/1000, ub, 'r.')
	plt.plot(np.linspace(addition_h, addition_l, (addition_h-addition_l)/5000+1).astype(int)/1000, lb, 'b.')
	plt.title('Price bounds on put options of {} + {} across strikes.'.format(st1, st2))
	plt.show()

if comb_type == 'P-':
	ub = []
	lb = []
	print('Finding constraints for P({}-{}) on date {}...'.format(st1, st2, expiration_date))

	for comb_strike in np.linspace(subtraction_l, subtraction_h, (subtraction_h-subtraction_l)/5000+1).astype(int):
		upper = math.inf
		lower = 0
		upper_k = math.inf
		lower_k = math.inf
		lower_type = ''

		standard = (opt1_df['Strike Price of the Option Times 1000'] >= min(opt2_df['Strike Price of the Option Times 1000']) + comb_strike) & \
			(opt1_df['Strike Price of the Option Times 1000'] <= max(opt2_df['Strike Price of the Option Times 1000']) + comb_strike)
		for k in opt1_df[standard & (opt1_df['C=Call, P=Put'] == 'P')]['Strike Price of the Option Times 1000']:
			filt1c = (opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == k)
			filt1p = (opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == k)
			filt2c = (opt2_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == k-comb_strike)
			filt2p = (opt2_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == k-comb_strike)
			if any(filt2c):
				upper_temp = float(opt1_df[filt1p]['Lowest  Closing Ask Across All Exchanges']) + float(opt2_df[filt2c]['Lowest  Closing Ask Across All Exchanges'])
				if upper_temp < upper:
					upper = float("{0:.2f}".format(upper_temp))
					upper_k = k
				lower_temp = float(opt2_df[filt2c]['Highest Closing Bid Across All Exchanges']) - float(opt1_df[filt1c]['Lowest  Closing Ask Across All Exchanges'])
				if lower_temp > lower:
					lower = float("{0:.2f}".format(lower_temp))
					lower_k = k
					lower_type = 'C'
			if any(filt2p):
				lower_temp = float(opt1_df[filt1p]['Highest Closing Bid Across All Exchanges']) - float(opt2_df[filt2p]['Lowest  Closing Ask Across All Exchanges'])
				if lower_temp > lower:
					lower = float("{0:.2f}".format(lower_temp))
					lower_k = k
					lower_type = 'P'
		ub.append(upper)
		lb.append(lower)
		print('{} <= P({}-{}, {}) <= {}'.format(lower, st1, st2, comb_strike, upper))
		if lower_type == '':
			print('Lower bound 0: by definition of an option.')
		elif lower_type == 'C':
			print('Lower bound {}: C({}, {}) = {} - C({}, {}) = {}'.format(lower, st2, lower_k-comb_strike, \
				float(opt2_df[(opt2_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == lower_k-comb_strike)]['Highest Closing Bid Across All Exchanges']), \
				st1, lower_k, \
				float(opt1_df[(opt1_df['C=Call, P=Put'] == 'C') & (opt1_df['Strike Price of the Option Times 1000'] == lower_k)]['Lowest  Closing Ask Across All Exchanges'])))
		else:
			print('Lower bound {}: P({}, {}) = {} - P({}, {}) = {}'.format(lower, st1, lower_k, \
				float(opt1_df[(opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == lower_k)]['Highest Closing Bid Across All Exchanges']), \
				st2, lower_k-comb_strike, \
				float(opt2_df[(opt2_df['C=Call, P=Put'] == 'P') & (opt2_df['Strike Price of the Option Times 1000'] == lower_k-comb_strike)]['Lowest  Closing Ask Across All Exchanges'])))
		
		print('Upper bound {}: P({}, {}) = {} + C({}, {}) = {}'.format(upper, st1, upper_k, \
			float(opt1_df[(opt1_df['C=Call, P=Put'] == 'P') & (opt1_df['Strike Price of the Option Times 1000'] == upper_k)]['Lowest  Closing Ask Across All Exchanges']), \
			st2, upper_k-comb_strike, \
			float(opt2_df[(opt2_df['C=Call, P=Put'] == 'C') & (opt2_df['Strike Price of the Option Times 1000'] == upper_k-comb_strike)]['Lowest  Closing Ask Across All Exchanges'])))
		print('')

	plt.plot(np.linspace(subtraction_l, subtraction_h, (subtraction_h-subtraction_l)/5000+1).astype(int)/1000, ub, 'r.')
	plt.plot(np.linspace(subtraction_l, subtraction_h, (subtraction_h-subtraction_l)/5000+1).astype(int)/1000, lb, 'b.')
	plt.title('Price bounds on put options of {} - {} across strikes.'.format(st1, st2))
	plt.show()

