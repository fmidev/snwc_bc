import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import time
import sys,os,getopt
import matplotlib.pyplot as plt
import seaborn as sns
import math
from XGBmodify import modify
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn


def density_scatter( x , y, ax = None, sort = True, bins = 20, yla = '', xla = '', picname='scatterplot',  **kwargs)   :
	"""
	code originally from
	https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density/53865762#53865762
	"""

	"""
	Scatter plot colored by 2d histogram
	"""
	if ax is None :
		fig , ax = plt.subplots()
	data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
	z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

	#To be sure to plot all data
	z[np.where(np.isnan(z))] = 0.0

	# Sort the points by density, so that the densest points are plotted last
	if sort :
		idx = z.argsort()
		x, y, z = x[idx], y[idx], z[idx]

	ax.scatter( x, y, c=z, **kwargs )
	lineStart = x.min()
	lineEnd = x.max()
	#add diagonal
	ax.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
	plt.xlabel(xla)
	plt.ylabel(yla)

	norm = Normalize(vmin = np.min(z), vmax = np.max(z))
	cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
	cbar.ax.set_ylabel('Density')
	plt.savefig(picname + '.png')
	return ax


def main():
	options, remainder = getopt.getopt(sys.argv[1:],[],['param=','path=','model1=','model2=','help'])

	for opt, arg in options:
		if opt == '--help':
			print('xgb_test_results.py param=T2m, WS, WG or RH, path=dir to test data, model1 and 2 are paths to the .joblib models to compare')
			exit()
		elif opt == '--param':
			param = arg
		elif opt == '--path':
			path = arg
		elif opt == '--model1':
			model1 = arg
		elif opt == '--model2':
			model2 = arg


	print(param)

	# read in the data and the models
	ds1 = pd.read_feather(path + 'utcmnwc2022q12.ftr', columns=None, use_threads=True);
	ds2 = pd.read_feather(path + 'utcmnwc2022q34.ftr', columns=None, use_threads=True);

	# subset five last days/month in 2022 as test set 
	df_temp = pd.concat([ds1,ds1])
	df_temp['origtime'] = df_temp['validdate'] - pd.to_timedelta(df_temp['leadtime'], unit='H')
	df_temp['day'] = df_temp['origtime'].dt.day
	# mask days 25-30 as test set
	mask = (df_temp['day'] >= 25) & (df_temp['day'] <= 30)

	test_set = df_temp[mask]

        # Drop the 'day_of_month' column if no longer needed
	test_set = test_set.drop(columns=['day','origtime'])

	del ds1, ds2

	data = modify(test_set,param)

	ds = pd.read_feather(path + 'utcmnwc2023q3.ftr', columns=None, use_threads=True);
	ds_tmp = modify(ds,param)
	data = pd.concat([data,ds_tmp])
	del ds, ds_tmp

	regressor1 = joblib.load(model1)
	regressor2 = joblib.load(model2)

	df_res = data
	#print(data.columns)

	# remove 0h
	data = data[data.leadtime != 0]
	data = data[data.leadtime != 1]
	df_res = df_res[df_res.leadtime != 0]
	df_res = df_res[df_res.leadtime != 1]
	# set the target parameter (F-O difference/model error)
	print("data columns:", data.columns)
	y_test = data.iloc[:,12]

	# remove columns not needed for training
	if param == 'T2m':
		remove = ['SID','validdate','TA_PT1M_AVG', 'Tero','T0bias']
	elif param == 'WS':
		remove = ['SID','validdate','WS_PT10M_AVG', 'WSero','WS0bias']
	elif param == 'RH':
		remove = ['SID','validdate','RH_PT1M_AVG', 'RHero','RH0bias']
	elif param == 'WG':
		remove = ['SID','validdate','WG_PT1H_MAX', 'WGero','WG0bias']
	# remove columns not needed to calculate the XGB model results
	X_test = data.drop(remove, axis=1)
	X_testold = X_test.drop(columns=['obs_elevation'])
	# make predictions for xgb (regressor) versions
	y_tsP1 = regressor1.predict(X_testold)
	y_tsP2 = regressor2.predict(X_test)

	print('Old model error:', mean_squared_error(y_test,y_tsP1,squared=False))
	print('New model error:', mean_squared_error(y_test,y_tsP2,squared=False))
	if param == 'WS':
		print('Raw model test error:', mean_squared_error(df_res['S10M'],df_res['WS_PT10M_AVG'],squared=False))
		xx = df_res['WS_PT10M_AVG'].values
		yy = df_res['S10M'].values
	elif param == 'T2m':
		print('Raw model test error:', mean_squared_error(df_res['T2M'],df_res['TA_PT1M_AVG'],squared=False))
		xx = df_res['TA_PT1M_AVG'].values
		yy = df_res['T2M'].values
	elif param == 'RH':
		print('Raw model test error:', mean_squared_error(df_res['RH2M'],df_res['RH_PT1M_AVG'],squared=False))
		xx = df_res['RH_PT1M_AVG'].values
		yy = df_res['RH2M'].values
	elif param == 'WG':
		print('Raw model test error:', mean_squared_error(df_res['GMAX'],df_res['WG_PT1H_MAX'],squared=False))
		xx = df_res['WG_PT1H_MAX'].values
		yy = df_res['GMAX'].values

	df_res['bc1'] = y_tsP1
	df_res['bc2'] = y_tsP2

	#yyy = df.loc[:,['TA_PT1M_AVG']].values # model data
	#xxx = df.loc[:,['T2M']].values #-y_ts obs
	print(xx[1:5])
	print(yy[1:5])

	density_scatter(xx, yy, bins = [60,60],yla= 'MNWC',xla='Observed',picname = 'mnwc_' + param)
	yyy = yy - df_res['bc1'].values
	density_scatter(xx, yyy, bins = [60,60],yla= 'XGB1',xla='Observed',picname = 'xgb1_' + param)
	yyy = yy - df_res['bc2'].values
	density_scatter(xx, yyy, bins = [60,60],yla= 'XGB2',xla='Observed',picname = 'xgb2_' + param)
	exit()

	ajat = sorted(df_res['leadtime'].unique().tolist())
	print(ajat)
	print("df_res columns:",df_res.columns)
	raw = [None] * len(ajat)
	xgb1 = [None] * len(ajat)
	xgb2 = [None] * len(ajat)
	for i in range(0,len(ajat)):
		print(ajat[i])
		tmp = df_res[df_res['leadtime']==ajat[i]] # all the parameter
		y_tmp = tmp.iloc[:,12] #.values #RHero --> mallivirhe
		print(y_tmp)
		y_xgb1 = tmp['bc1']
		y_xgb2 = tmp['bc2']
		xgb1[i] = mean_squared_error(y_tmp,y_xgb1,squared=False)
		xgb2[i] = mean_squared_error(y_tmp,y_xgb2,squared=False)
		if param == 'WS':
			raw[i] = mean_squared_error(tmp['S10M'],(tmp['S10M'].values-y_tmp),squared=False)
		elif param == 'T2m':
			raw[i] = mean_squared_error(tmp['T2M'],(tmp['T2M'].values-y_tmp),squared=False)
		elif param == 'RH':
			raw[i] = mean_squared_error(tmp['RH2M'],(tmp['RH2M'].values-y_tmp),squared=False)
		elif param == 'WG':
			raw[i] = mean_squared_error(tmp['GMAX'],(tmp['GMAX'].values-y_tmp),squared=False)

	print(raw)
	print(xgb1)
	print(xgb2)
	#print(old)

	plt.figure(2)
	plt.plot(ajat, raw, 'r', label="mnwc")
	plt.plot(ajat, xgb1, 'g', label="XGB1")
	plt.plot(ajat, xgb2, 'b', label="XGB2")
	plt.title("NWC RMSE")
	plt.xlabel("leadtime")
	plt.ylabel("RMSE")
	plt.legend(loc="lower right")
	plt.grid()
	if param == 'T2m':
		plt.ylim(0,2)
	elif param == 'WS':
		plt.ylim(0,2)
	elif param == 'WG':
		plt.ylim(0,3)
	elif param == 'RH':
		plt.ylim(0,12)
	plt.savefig('leadt_' + param + '_rmse.png')

#plt.show()
if __name__ == "__main__":
        main()
