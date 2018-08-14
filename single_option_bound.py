import pdb
import sys
import os
import numpy as np
import pandas as pd
import datetime, time
import math
from gurobipy import *

# Given the current price of Option(ST, K, T) in the market
# Return the pricing bound of Option(ST) on any strike k and any expiration date t
input_dir = sys.argv[1]
price_date = sys.argv[2]
assert(sys.argv[3] == 'C' or sys.argv[3] == 'P'), "option type should either be C or P."
if sys.argv[3] == 'C':
	call_or_put = 1
else:
	call_or_put = -1
st = sys.argv[4]
days_to_expiration = int(sys.argv[5])
# strike_on_expiration = int(sys.argv[6])
msft = 85540

opt_stats_filename = os.path.join(input_dir, st+"_"+price_date+".xlsx")
opt_df = pd.read_excel(opt_stats_filename)
assert(np.array(opt_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day."

# Preprocessing - convert puts to calls
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
# expiration date range
# date_strike_pairs = np.array(sorted(opt_df['Expiration Date of the Option'].value_counts().items()))
# dates = date_strike_pairs[:, 0]
# pdb.set_trace()
# strike_l, h = stats.t.interval(0.68, len(option_df['Strike Price of the Op	print('############################C({}{}+{}{}, {})############################'.format(coeff1, st1, coeff2, st2, strike))
# strike_l = np.ceil(strike_l/500)*500
# l, strike_h = stats.t.interval(0.95, len(option_df['Strike Price of the Option Times 1000'])-1, loc=np.mean(option_df['Strike Price of the Option Times 1000']), \
# 	scale=np.std(option_df['Strike Price of the Option Times 1000']))
# strike_h = np.floor(strike_h/500)*500
# strikes = np.linspace(strike_l, strike_h+10000, (strike_h+10000-strike_l)/500+1)
# option_df = option_df[(option_df['Strike Price of the Option Times 1000']>=strike_l) & (option_df['Strike Price of the Option Times 1000']<=strike_h)]
strike_low = 50000
strike_high = 110000
for strike_on_expiration in np.linspace(strike_low, strike_high, (strike_high-strike_low)/500+1):
	print('############################{}({}, {}, {})############################'.format(sys.argv[3], st, strike_on_expiration, days_to_expiration))
	# Calculating the upper bound
	try:
		model = Model("ub")
		model.setParam('OutputFlag', False)
		alpha = model.addVars(1, len(opt_df))
		if call_or_put == 1:
			model.addLConstr(sum(alpha[0,i]*opt_df['Unit'][i] for i in range(0, len(opt_df))), GRB.GREATER_EQUAL, call_or_put)
		else:
			model.addLConstr(sum(alpha[0,i]*opt_df['Unit'][i] for i in range(0, len(opt_df))), GRB.LESS_EQUAL, call_or_put)	
		model.addLConstr(sum(alpha[0,i]*opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(opt_df))), \
			GRB.LESS_EQUAL, strike_on_expiration*sum(alpha[0,i]*opt_df['Unit'][i] for i in range(0, len(opt_df))))
		model.addLConstr(sum(alpha[0,i]*opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(opt_df))), \
			GRB.LESS_EQUAL, strike_on_expiration*call_or_put)
		for i in range(0, len(opt_df)):
			model.addLConstr(alpha[0,i]*opt_df['Expiration Date of the Option'][i], GRB.GREATER_EQUAL, alpha[0,i]*days_to_expiration)
		model.setObjective(sum(alpha[0,i]*opt_df['Lowest  Closing Ask Across All Exchanges'][i] for i in range(0, len(opt_df))), GRB.MINIMIZE)
		model.optimize()
		# for v in model.getVars():
		# 	print('%s %g' % (v.varName, v.x))
		# print('Obj: %g' % model.objVal)
	except GurobiError as e:
		print('Error code ' + str(e.errno) + ": " + str(e))
	except AttributeError:
		print('Encountered an attribute error')

	expense = 0
	payoff = 0
	for i in range(0, len(opt_df)):
		if alpha[0,i].x > 1e-4:
			print("Buy {} fraction of {}({}, {}, {}) with ask price {}".format(float("{0:.4f}".format(alpha[0,i].x)), \
				opt_df['C=Call, P=Put'][i], st, np.abs(opt_df['Strike Price of the Option Times 1000'][i]), opt_df['Expiration Date of the Option'][i], \
				opt_df['Lowest  Closing Ask Across All Exchanges'][i]))
			expense = expense + alpha[0,i].x*opt_df['Lowest  Closing Ask Across All Exchanges'][i]
			payoff = payoff + alpha[0,i].x*max(msft*opt_df['Unit'][i]-opt_df['Strike Price of the Option Times 1000'][i],0)
	print('~~~~~~~The upper bound is {}. The realized payoff is {} - {}~~~~~~~'.format(float("{0:.2f}".format(expense)), payoff, max((msft-strike_on_expiration)*call_or_put, 0)))

	# Calculating the lower bound
	try:
		model = Model("lb")
		model.setParam('OutputFlag', False)
		model.setParam('IntFeasTol', 1e-9)
		beta = model.addVars(1, len(opt_df))
		binary = model.addVars(1, len(opt_df), vtype=GRB.BINARY)
		gamma = model.addVars(1, len(opt_df))
		model.addLConstr(sum(binary[0,i] for i in range(0, len(opt_df))), GRB.LESS_EQUAL, 1)
		for i in range(0, len(opt_df)):
			model.addLConstr(beta[0,i]-1000*binary[0,i], GRB.LESS_EQUAL, 0)
		if call_or_put == 1:
			model.addLConstr(sum(beta[0,i]*opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(opt_df))), GRB.GREATER_EQUAL, 0)
			model.addLConstr(sum((beta[0,i]-gamma[0,i])*opt_df['Unit'][i] for i in range(0, len(opt_df))), GRB.LESS_EQUAL, call_or_put)
			model.addLConstr(sum((beta[0,i]-gamma[0,i])*opt_df['Unit'][i] for i in range(0, len(opt_df))), GRB.GREATER_EQUAL, 0)
		else:
			model.addLConstr(sum(beta[0,i]*opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(opt_df))), GRB.LESS_EQUAL, 0)
			model.addLConstr(sum((beta[0,i]-gamma[0,i])*opt_df['Unit'][i] for i in range(0, len(opt_df))), GRB.GREATER_EQUAL, call_or_put)
			model.addLConstr(sum((beta[0,i]-gamma[0,i])*opt_df['Unit'][i] for i in range(0, len(opt_df))), GRB.LESS_EQUAL, 0)
		model.addLConstr(sum((beta[0,i]-gamma[0,i]) for i in range(0, len(opt_df)))*call_or_put*strike_on_expiration + \
			sum(gamma[0,i]*opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(opt_df))), \
			GRB.LESS_EQUAL, sum(beta[0,i]*opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(opt_df))))
		model.addLConstr(call_or_put*strike_on_expiration+sum(gamma[0,i]*opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(opt_df))), \
			GRB.LESS_EQUAL, sum(beta[0,i]*opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(opt_df))))
		for i in range(0, len(opt_df)):
			model.addLConstr(beta[0,i]*opt_df['Expiration Date of the Option'][i], GRB.LESS_EQUAL, beta[0,i]*days_to_expiration)
			model.addLConstr(gamma[0,i]*opt_df['Expiration Date of the Option'][i], GRB.GREATER_EQUAL, gamma[0,i]*days_to_expiration)
		model.setObjective(sum(beta[0,i]*opt_df['Highest Closing Bid Across All Exchanges'][i] for i in range(0, len(opt_df))) - \
			sum(gamma[0,i]*opt_df['Lowest  Closing Ask Across All Exchanges'][i] for i in range(0, len(opt_df))), GRB.MAXIMIZE)
		model.optimize()
		# for v in model.getVars():
		# 	print('%s %g' % (v.varName, v.x))
		# print('Obj: %g' % model.objVal)
	except GurobiError as e:
		print('Error code ' + str(e.errno) + ": " + str(e))
	except AttributeError:
		print('Encountered an attribute error')

	profit = 0
	payoff = 0
	for i in range(0, len(opt_df)):
		if beta[0,i].x > 1e-4:
			print("Sell {} fraction of {}({}, {}, {}) with bid price {}".format(float("{0:.4f}".format(beta[0,i].x)), \
				opt_df['C=Call, P=Put'][i], st, np.abs(opt_df['Strike Price of the Option Times 1000'][i]), opt_df['Expiration Date of the Option'][i], \
				opt_df['Highest Closing Bid Across All Exchanges'][i]))
			profit = profit + beta[0,i].x*opt_df['Highest Closing Bid Across All Exchanges'][i]
			payoff = payoff - beta[0,i].x*max(msft*opt_df['Unit'][i]-opt_df['Strike Price of the Option Times 1000'][i],0)
		if gamma[0,i].x > 1e-4:
			print("Buy {} fraction of {}({}, {}, {}) with ask price {}".format(float("{0:.4f}".format(gamma[0,i].x)), \
				opt_df['C=Call, P=Put'][i], st, np.abs(opt_df['Strike Price of the Option Times 1000'][i]), opt_df['Expiration Date of the Option'][i], \
				opt_df['Lowest  Closing Ask Across All Exchanges'][i]))
			profit = profit - gamma[0,i].x*opt_df['Lowest  Closing Ask Across All Exchanges'][i]
			payoff = payoff + gamma[0,i].x*max(msft*opt_df['Unit'][i]-opt_df['Strike Price of the Option Times 1000'][i],0)
	print('~~~~~~~The lower bound is {}. The realized payoff is {} - {}~~~~~~~'.format(float("{0:.2f}".format(profit)), max((msft-strike_on_expiration)*call_or_put,0), np.abs(payoff)))


# payoff version
# try:
# 	m = Model("payoff_version")
# 	m.setParam('OutputFlag', False)
# 	alph = m.addVars(1, len(opt_df))
# 	s_low = 5000
# 	s_high = 500000
# 	for s in np.linspace(s_low, s_high, (s_high-s_low)/500+1):
# 		m.addLConstr(sum(alph[0,i]*max(s*opt_df['Unit'][i]-opt_df['Strike Price of the Option Times 1000'][i], 0) for i in range(0, len(opt_df))), \
# 			GRB.GREATER_EQUAL, max((s-strike_on_expiration)*call_or_put, 0))
# 	for i in range(0, len(opt_df)):
# 		m.addLConstr(alph[0,i]*opt_df['Expiration Date of the Option'][i], GRB.GREATER_EQUAL, alph[0,i]*days_to_expiration)
# 	m.setObjective(sum(alph[0,i]*opt_df['Lowest  Closing Ask Across All Exchanges'][i] for i in range(0, len(opt_df))), GRB.MINIMIZE)
# 	m.optimize()
# 	# for v in m.getVars():
# 	# 	print('%s %g' % (v.varName, v.x))
# 	# print('Obj: %g' % m.objVal)
# except GurobiError as e:
# 	print('Error code ' + str(e.errno) + ": " + str(e))
# except AttributeError:
# 	print('Encountered an attribute error')
# exp = 0
# pay = 0
# for i in range(0, len(opt_df)):
# 	if alph[0, i].x > 1e-4:
# 		print("Buy {} fraction of {}({}, {}, {}) with ask price {}".format(float("{0:.4f}".format(alph[0, i].x)), \
# 			opt_df['C=Call, P=Put'][i], st, np.abs(opt_df['Strike Price of the Option Times 1000'][i]), opt_df['Expiration Date of the Option'][i], \
# 			opt_df['Lowest  Closing Ask Across All Exchanges'][i]))
# 		exp = exp + alph[0, i].x*opt_df['Lowest  Closing Ask Across All Exchanges'][i]
# 		pay = pay + alph[0, i].x*max(msft*opt_df['Unit'][i]-opt_df['Strike Price of the Option Times 1000'][i],0)
# print('~~~~~~~The upper bound is {}. The realized payoff is {} - {}~~~~~~~'.format(float("{0:.2f}".format(exp)), pay, max((msft-strike_on_expiration)*call_or_put, 0)))