import pdb
import sys
import os
import numpy as np
import pandas as pd
from gurobipy import *

# input from command
input_dir = sys.argv[1]
price_date = sys.argv[2]
buy_or_sell = sys.argv[3]
coeff1 = int(sys.argv[4])
st1 = sys.argv[5]
coeff2 = int(sys.argv[6])
st2 = sys.argv[7]
#strike = int(sys.argv[8])
#price = float(sys.argv[8])
expiration_date = int(sys.argv[8])
comb_type = sys.argv[9]
opt1_stats_filename = os.path.join(input_dir, st1+"_"+price_date+".xlsx")
opt1_df = pd.read_excel(opt1_stats_filename)
opt2_stats_filename = os.path.join(input_dir, st2+"_"+price_date+".xlsx")
opt2_df = pd.read_excel(opt2_stats_filename)

# current price 
goog = 1053400
msft = 85540

assert(np.array(opt1_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"
assert(np.array(opt2_df['AM Settlement Flag'].astype(int)).all()==0), "options on the security expire at the market open of the last trading day"

# preprocessing - convert all puts to calls
# TO-DO: maybe also include options with bid at 0 ???
# opt1_df = opt1_df[(opt1_df['Expiration Date of the Option']==expiration_date) & (opt1_df['Highest Closing Bid Across All Exchanges'] > 0)]
# opt2_df = opt2_df[(opt2_df['Expiration Date of the Option']==expiration_date) & (opt2_df['Highest Closing Bid Across All Exchanges'] > 0)]
opt1_df = opt1_df[(opt1_df['Expiration Date of the Option']==expiration_date)]
opt2_df = opt2_df[(opt2_df['Expiration Date of the Option']==expiration_date)]
opt1_df['Strike Price of the Option Times 1000'] = opt1_df['Strike Price of the Option Times 1000'].astype(int)
# opt1_df.loc[opt1_df['C=Call, P=Put'] == 'P', 'Strike Price of the Option Times 1000'] = -1 * opt1_df[opt1_df['C=Call, P=Put'] == 'P']['Strike Price of the Option Times 1000']
opt1_df['Highest Closing Bid Across All Exchanges'] = opt1_df['Highest Closing Bid Across All Exchanges'].astype(float)
opt1_df['Lowest  Closing Ask Across All Exchanges'] = opt1_df['Lowest  Closing Ask Across All Exchanges'].astype(float)

opt2_df['Strike Price of the Option Times 1000'] = opt2_df['Strike Price of the Option Times 1000'].astype(int)
# opt2_df.loc[opt2_df['C=Call, P=Put'] == 'P', 'Strike Price of the Option Times 1000'] = -1 * opt2_df[opt2_df['C=Call, P=Put'] == 'P']['Strike Price of the Option Times 1000']
opt2_df['Highest Closing Bid Across All Exchanges'] = opt2_df['Highest Closing Bid Across All Exchanges'].astype(float)
opt2_df['Lowest  Closing Ask Across All Exchanges'] = opt2_df['Lowest  Closing Ask Across All Exchanges'].astype(float)
opt1_df.sort_values(by = ['Strike Price of the Option Times 1000'], inplace = True)
opt2_df.sort_values(by = ['Strike Price of the Option Times 1000'], inplace = True)

# interested strike range
addition_l = coeff1*min(opt1_df['Strike Price of the Option Times 1000']) + coeff2*min(opt2_df['Strike Price of the Option Times 1000'])
addition_h = coeff1*max(opt1_df['Strike Price of the Option Times 1000']) + coeff2*max(opt2_df['Strike Price of the Option Times 1000'])
for strike in np.linspace(addition_l, addition_h, (addition_h-addition_l)/500+1).astype(int):
	print('############################C({}{}+{}{}, {})############################'.format(coeff1, st1, coeff2, st2, strike))
	buy_or_sell = 'buy'
	price = 10000
	if buy_or_sell == 'buy':
		# The exchange would sell an item with intrinsic low value at a higher price and cover with buying an item with intrinsic high value at a lower price.
		opt1_df_buy = opt1_df[opt1_df['C=Call, P=Put']=='C']
		opt2_df_buy = opt2_df[opt2_df['C=Call, P=Put']=='C']
		opt1_asks = np.array(opt1_df_buy['Lowest  Closing Ask Across All Exchanges'])
		opt2_asks = np.array(opt2_df_buy['Lowest  Closing Ask Across All Exchanges'])
		asks = np.concatenate([opt1_asks, opt2_asks])
		opt1_strikes = np.array(opt1_df_buy['Strike Price of the Option Times 1000'])
		opt2_strikes = np.array(opt2_df_buy['Strike Price of the Option Times 1000'])
		strikes = np.concatenate([opt1_strikes, opt2_strikes])
		try:
			model = Model("ub")
			model.setParam('OutputFlag', False)
			frac = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.CONTINUOUS, name="lambda", column=None)
			x = model.addVars(1, len(strikes))
			model.addLConstr(sum(x[0,i] for i in range(0, len(opt1_strikes))), GRB.GREATER_EQUAL, frac*coeff1)
			model.addLConstr(sum(x[0,i] for i in range(len(opt1_strikes), len(strikes))), GRB.GREATER_EQUAL, frac*coeff2)
			model.addLConstr(sum(x[0,i]*strikes[i] for i in range(0, len(strikes))), GRB.LESS_EQUAL, frac*strike)
			model.setObjective(frac*price-sum(x[0,i]*asks[i] for i in range(0, len(strikes))), GRB.MAXIMIZE)
			model.optimize()
			# for v in model.getVars():
			# 	print('%s %g' % (v.varName, v.x))
			# print('Obj: %g' % model.objVal)
		except GurobiError as e:
			print('Error code ' + str(e.errno) + ": " + str(e))
		except AttributeError:
			print('Encountered an attribute error')

		#print('The exchange sells by accepting {} fraction of the combinatorial bid order on call option of {}{}+{}{} with strike price {} at bid price {}'.format(frac.x, coeff1, st1, coeff2, st2, strike, price))
		#print('The exchange will cover the {} fraction of the combinatorial bid order with...'.format(frac.x))
		profit = frac.x*price
		payoff = 0
		for i in range(0, len(strikes)):
			if x[0, i].x > 1e-2:
				if i < len(opt1_asks):
					profit = profit-x[0, i].x*asks[i]
					print("Buy {} fraction of call option on {} at strike {} with ask price {}".format(float("{0:.4f}".format(x[0, i].x)), st1, strikes[i], asks[i]))
					payoff = payoff + x[0, i].x*max(goog-strikes[i], 0)
				else: 
					profit = profit-x[0, i].x*asks[i]
					print("Buy {} fraction of call option on {} at strike {} with ask price {}".format(float("{0:.4f}".format(x[0, i].x)), st2, strikes[i], asks[i]))
					payoff = payoff + x[0, i].x*max(msft-strikes[i], 0)
		#print('The maximized revenue is {}.'.format(profit))
		print('~~~~~~~The upper bound is {}, with realized payoff {}~~~~~~~'.format(float("{0:.2f}".format(frac.x*price-profit)), float("{0:.2f}".format(payoff))))

	buy_or_sell = 'sell'
	price = 0
	if buy_or_sell == 'sell':
		# The exchange would buy an item with intrinsic high value at a lower price and cover with selling an item with intrinsic lower value at a higher price.
		opt1_df_sell = opt1_df[opt1_df['C=Call, P=Put']=='C']
		opt2_df_sell = opt2_df[opt2_df['C=Call, P=Put']=='P']
		idx = 0
		F = []
		X = []
		Y= []
		obj = []
		for opt_df_sell in [opt1_df[opt1_df['C=Call, P=Put']=='C'], opt2_df[opt2_df['C=Call, P=Put']=='C']]:
			if idx == 0:
				opt_df_buy = opt2_df[opt2_df['C=Call, P=Put']=='P']
			else:
				opt_df_buy = opt1_df[opt1_df['C=Call, P=Put']=='P']
			opt1_bids = np.array(opt_df_sell['Highest Closing Bid Across All Exchanges'])
			opt2_asks = np.array(opt_df_buy['Lowest  Closing Ask Across All Exchanges'])
			opt1_strikes = np.array(opt_df_sell['Strike Price of the Option Times 1000'])
			opt2_strikes = np.array(opt_df_buy['Strike Price of the Option Times 1000'])
			strikes = np.concatenate([opt1_strikes, opt2_strikes])
			try:
			    # Create a new model
			    model = Model()
			    model.setParam('OutputFlag', False)
			    frac = model.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=GRB.CONTINUOUS, name="lambda", column=None)
			    x = model.addVars(1, len(opt1_strikes))
			    y = model.addVars(len(opt1_strikes), len(opt2_strikes))
			    if idx == 0:
			    	model.addLConstr(sum(x[0, i] for i in range(0, len(opt1_strikes))), GRB.LESS_EQUAL, frac*coeff1)
			    else:
			    	model.addLConstr(sum(x[0, i] for i in range(0, len(opt1_strikes))), GRB.LESS_EQUAL, frac*coeff2)

			    for i in range(0, len(opt1_strikes)):
			    	model.addLConstr(x[0, i]*opt1_strikes[i]+sum(y[i,j]*opt2_strikes[j] for j in range(0, len(opt2_strikes))), GRB.GREATER_EQUAL, x[0, i]*strike)
			    	if idx == 0:
			    		model.addLConstr(sum(y[i,j] for j in range(0, len(opt2_strikes))), GRB.LESS_EQUAL, x[0, i]*coeff2)
			    	else:
			    		model.addLConstr(sum(y[i,j] for j in range(0, len(opt2_strikes))), GRB.LESS_EQUAL, x[0, i]*coeff1)
			    model.setObjective(sum(x[0, i]*opt1_bids[i] for i in range(0, len(opt1_strikes)))-frac*price \
					-sum(sum(y[i,j] for i in range(0, len(opt1_strikes)))*opt2_asks[j] for j in range(0,len(opt2_strikes))), GRB.MAXIMIZE)
			    model.optimize()
				# for v in model.getVars():
				# 	print('%s %g' % (v.varName, v.x))
				# print('Obj: %g' % model.objVal)
			except GurobiError as e:
				print('Error code ' + str(e.errno) + ": " + str(e))
			except AttributeError:
				print('Encountered an attribute error')

			F.append(frac.x)
			x_temp = [x[0, i].x for i in range(0, len(opt1_strikes))]
			X.append(x_temp)
			y_temp = [sum(y[i, j].x for i in range(0, len(opt1_strikes))) for j in range(0, len(opt2_strikes))]
			Y.append(y_temp)
			obj.append(model.objVal)
			idx = idx + 1

		res = np.argmax(obj)
		res_frac = F[res]
		res_x = X[res]
		res_y = Y[res]
		# print('The exchange buys by accepting {} fraction of the combinatorial ask order on call option of {}{}+{}{} with strike price {} at ask price {}'.format(frac.x, coeff1, st1, coeff2, st2, strike, price))
		# print('The exchange will cover the {} fraction of the combinatorial ask order with...'.format(frac.x))
		if max(obj) != 0:
			profit = -1*res_frac*price
			payoff = 0
			if res == 0:
				opt_df_sell = opt1_df[opt1_df['C=Call, P=Put']=='C']
				opt_df_buy = opt2_df[opt2_df['C=Call, P=Put']=='P']
				opt1_bids = np.array(opt_df_sell['Highest Closing Bid Across All Exchanges'])
				opt2_asks = np.array(opt_df_buy['Lowest  Closing Ask Across All Exchanges'])
				opt1_strikes = np.array(opt_df_sell['Strike Price of the Option Times 1000'])
				opt2_strikes = np.array(opt_df_buy['Strike Price of the Option Times 1000'])
				strikes = np.concatenate([opt1_strikes, opt2_strikes])
				for i in range(0, len(opt1_strikes)):
					if res_x[i] > 1e-2:
						profit = profit+res_x[i]*opt1_bids[i]
						print("Sell {} fraction of call option on {} at strike {} with bid price {}".format(float("{0:.4f}".format(res_x[i])), st1, opt1_strikes[i], opt1_bids[i]))
						payoff = payoff - res_x[i]*max(goog-opt1_strikes[i], 0)
				for j in range(0, len(opt2_strikes)):
					if res_y[j] > 1e-2:
						profit = profit-res_y[j]*opt2_asks[j]
						print("Buy {} fraction of put option on {} at strike {} with ask price {}".format(float("{0:.4f}".format(res_y[j])), st2, opt2_strikes[j], opt2_asks[j]))
						payoff = payoff + res_y[j]*max(opt2_strikes[j]-msft, 0)
			else:
				opt_df_sell = opt2_df[opt2_df['C=Call, P=Put']=='C']
				opt_df_buy = opt1_df[opt1_df['C=Call, P=Put']=='P']
				opt1_bids = np.array(opt_df_sell['Highest Closing Bid Across All Exchanges'])
				opt2_asks = np.array(opt_df_buy['Lowest  Closing Ask Across All Exchanges'])
				opt1_strikes = np.array(opt_df_sell['Strike Price of the Option Times 1000'])
				opt2_strikes = np.array(opt_df_buy['Strike Price of the Option Times 1000'])
				strikes = np.concatenate([opt1_strikes, opt2_strikes])
				for i in range(0, len(opt1_strikes)):
					if res_x[i] > 1e-2:
						profit = profit+res_x[i]*opt1_bids[i]
						print("Sell {} fraction of call option on {} at strike {} with bid price {}".format(float("{0:.4f}".format(res_x[i])), st2, opt1_strikes[i], opt1_bids[i]))
						payoff = payoff - res_x[i]*max(msft-opt1_strikes[i], 0)
				for j in range(0, len(opt2_strikes)):
					if res_y[j] > 1e-2:
						profit = profit-res_y[j]*opt2_asks[j]
						print("Buy {} fraction of put option on {} at strike {} with ask price {}".format(float("{0:.4f}".format(res_y[j])), st1, opt2_strikes[j], opt2_asks[j]))
						payoff = payoff + res_y[j]*max(opt2_strikes[j]-goog, 0)
			print('~~~~~~~The lower bound is {}, with realized payoff {}~~~~~~~'.format(float("{0:.2f}".format(profit)), float("{0:.2f}".format(payoff))))
		else:
			print('~~~~~~~The lower bound is {}, with realized payoff {}~~~~~~~'.format(float("{0:.2f}".format(0)), float("{0:.2f}".format(payoff))))


pdb.set_trace()