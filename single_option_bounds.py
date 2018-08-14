import pdb
import sys
import os
import numpy as np
import pandas as pd
import datetime, time
import math
from gurobipy import *

class SingleOptionBounds:
	def __init__(self, st, opt_df, strike_low, strike_high, strike_interval, days_to_expiration):
		self.st = st
		self.opt_df = opt_df
		self.strike_low = strike_low
		self.strike_high = strike_high
		self.strike_interval = strike_interval
		self.days_to_expiration = days_to_expiration

	def tighten_spread(self):
		strikes = []
		call_bids = []
		call_asks = []
		put_bids = []
		put_asks = []
		for strike_on_expiration in np.linspace(self.strike_low, self.strike_high, (self.strike_high-self.strike_low)/self.strike_interval+1):
			strikes.append(strike_on_expiration)
			for c_or_p in ['C', 'P']:
				print('############################{}({}, {}, {})############################'.format(c_or_p, self.st, strike_on_expiration, self.days_to_expiration))
				if c_or_p == 'C':
					call_or_put = 1
				elif c_or_p == 'P':
					call_or_put = -1

				try:
					model = Model("ub")
					model.setParam('OutputFlag', False)
					alpha = model.addVars(1, len(self.opt_df))
					if call_or_put == 1:
						model.addLConstr(sum(alpha[0,i]*self.opt_df['Unit'][i] for i in range(0, len(self.opt_df))), GRB.GREATER_EQUAL, call_or_put)
					else:
						model.addLConstr(sum(alpha[0,i]*self.opt_df['Unit'][i] for i in range(0, len(self.opt_df))), GRB.LESS_EQUAL, call_or_put)	
					model.addLConstr(sum(alpha[0,i]*self.opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(self.opt_df))), \
						GRB.LESS_EQUAL, strike_on_expiration*sum(alpha[0,i]*self.opt_df['Unit'][i] for i in range(0, len(self.opt_df))))
					model.addLConstr(sum(alpha[0,i]*self.opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(self.opt_df))), \
						GRB.LESS_EQUAL, strike_on_expiration*call_or_put)
					for i in range(0, len(self.opt_df)):
						model.addLConstr(alpha[0,i]*self.opt_df['Expiration Date of the Option'][i], GRB.GREATER_EQUAL, alpha[0,i]*self.days_to_expiration)
					model.setObjective(sum(alpha[0,i]*self.opt_df['Lowest  Closing Ask Across All Exchanges'][i] for i in range(0, len(self.opt_df))), GRB.MINIMIZE)
					model.optimize()
				except GurobiError as e:
					print('Error code ' + str(e.errno) + ": " + str(e))
				except AttributeError:
					print('Encountered an attribute error')

				expense = 0
				# payoff = 0
				for i in range(0, len(self.opt_df)):
					if alpha[0,i].x > 1e-4:
						print("Buy {} fraction of {}({}, {}, {}) with ask price {}".format(float("{0:.4f}".format(alpha[0,i].x)), \
							self.opt_df['C=Call, P=Put'][i], self.st, np.abs(self.opt_df['Strike Price of the Option Times 1000'][i]), self.opt_df['Expiration Date of the Option'][i], \
							self.opt_df['Lowest  Closing Ask Across All Exchanges'][i]))
						expense = expense + alpha[0,i].x*self.opt_df['Lowest  Closing Ask Across All Exchanges'][i]
						# payoff = payoff + alpha[0,i].x*max(msft*self.opt_df['Unit'][i]-self.opt_df['Strike Price of the Option Times 1000'][i],0)
				print('~~~~~~~The upper bound is {}.~~~~~~~'.format(float("{0:.2f}".format(expense))))

				try:
					model = Model("lb")
					model.setParam('OutputFlag', False)
					model.setParam('IntFeasTol', 1e-9)
					beta = model.addVars(1, len(self.opt_df))
					binary = model.addVars(1, len(self.opt_df), vtype=GRB.BINARY)
					gamma = model.addVars(1, len(self.opt_df))
					model.addLConstr(sum(binary[0,i] for i in range(0, len(self.opt_df))), GRB.LESS_EQUAL, 1)
					for i in range(0, len(self.opt_df)):
						model.addLConstr(beta[0,i]-1000*binary[0,i], GRB.LESS_EQUAL, 0)
					if call_or_put == 1:
						model.addLConstr(sum(beta[0,i]*self.opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(self.opt_df))), GRB.GREATER_EQUAL, 0)
						model.addLConstr(sum((beta[0,i]-gamma[0,i])*self.opt_df['Unit'][i] for i in range(0, len(self.opt_df))), GRB.LESS_EQUAL, call_or_put)
						model.addLConstr(sum((beta[0,i]-gamma[0,i])*self.opt_df['Unit'][i] for i in range(0, len(self.opt_df))), GRB.GREATER_EQUAL, 0)
					else:
						model.addLConstr(sum(beta[0,i]*self.opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(self.opt_df))), GRB.LESS_EQUAL, 0)
						model.addLConstr(sum((beta[0,i]-gamma[0,i])*self.opt_df['Unit'][i] for i in range(0, len(self.opt_df))), GRB.GREATER_EQUAL, call_or_put)
						model.addLConstr(sum((beta[0,i]-gamma[0,i])*self.opt_df['Unit'][i] for i in range(0, len(self.opt_df))), GRB.LESS_EQUAL, 0)
					model.addLConstr(sum((beta[0,i]-gamma[0,i]) for i in range(0, len(self.opt_df)))*call_or_put*strike_on_expiration + \
						sum(gamma[0,i]*self.opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(self.opt_df))), \
						GRB.LESS_EQUAL, sum(beta[0,i]*self.opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(self.opt_df))))
					model.addLConstr(sum(binary[0, i] for i in range(0, len(self.opt_df)))*call_or_put*strike_on_expiration+sum(gamma[0,i]*self.opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(self.opt_df))), \
						GRB.LESS_EQUAL, sum(beta[0,i]*self.opt_df['Strike Price of the Option Times 1000'][i] for i in range(0, len(self.opt_df))))
					for i in range(0, len(self.opt_df)):
						model.addLConstr(beta[0,i]*self.opt_df['Expiration Date of the Option'][i], GRB.LESS_EQUAL, beta[0,i]*self.days_to_expiration)
						model.addLConstr(gamma[0,i]*self.opt_df['Expiration Date of the Option'][i], GRB.GREATER_EQUAL, gamma[0,i]*self.days_to_expiration)
					model.setObjective(sum(beta[0,i]*self.opt_df['Highest Closing Bid Across All Exchanges'][i] for i in range(0, len(self.opt_df))) - \
						sum(gamma[0,i]*self.opt_df['Lowest  Closing Ask Across All Exchanges'][i] for i in range(0, len(self.opt_df))), GRB.MAXIMIZE)
					model.optimize()
				except GurobiError as e:
					print('Error code ' + str(e.errno) + ": " + str(e))
				except AttributeError:
					print('Encountered an attribute error')

				profit = 0
				# payoff = 0
				for i in range(0, len(self.opt_df)):
					if beta[0,i].x > 1e-4:
						print("Sell {} fraction of {}({}, {}, {}) with bid price {}".format(float("{0:.4f}".format(beta[0,i].x)), \
							self.opt_df['C=Call, P=Put'][i], self.st, np.abs(self.opt_df['Strike Price of the Option Times 1000'][i]), self.opt_df['Expiration Date of the Option'][i], \
							self.opt_df['Highest Closing Bid Across All Exchanges'][i]))
						profit = profit + beta[0,i].x*self.opt_df['Highest Closing Bid Across All Exchanges'][i]
						# payoff = payoff - beta[0,i].x*max(msft*self.opt_df['Unit'][i]-self.opt_df['Strike Price of the Option Times 1000'][i],0)
					if gamma[0,i].x > 1e-4:
						print("Buy {} fraction of {}({}, {}, {}) with ask price {}".format(float("{0:.4f}".format(gamma[0,i].x)), \
							self.opt_df['C=Call, P=Put'][i], self.st, np.abs(self.opt_df['Strike Price of the Option Times 1000'][i]), self.opt_df['Expiration Date of the Option'][i], \
							self.opt_df['Lowest  Closing Ask Across All Exchanges'][i]))
						profit = profit - gamma[0,i].x*self.opt_df['Lowest  Closing Ask Across All Exchanges'][i]
						# payoff = payoff + gamma[0,i].x*max(msft*self.opt_df['Unit'][i]-self.opt_df['Strike Price of the Option Times 1000'][i],0)
				print('~~~~~~~The lower bound is {}.~~~~~~~'.format(float("{0:.2f}".format(profit))))
				assert(expense >= 0), "Upper bound falls negative."
				assert(profit >= 0), "Lower bound falls negative."
				assert(expense >= profit), "Arbitrage found."
				if call_or_put == 1:
					call_asks.append(expense)
					call_bids.append(profit)
				else:
					put_asks.append(expense)
					put_bids.append(profit)
		return np.array(strikes), np.array(call_bids), np.array(call_asks), np.array(put_bids), np.array(put_asks)