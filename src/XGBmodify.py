import pandas as pd
import numpy as np
import time
import sys
import math
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# modify time to sin/cos representation
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

# data columns listed
# 'index', 'leadtime', 'SID', 'lat', 'lon', 'model_elevation',
#       'validdate', 'fcdate', 'D10M', 'T2M', 'CCTOT', 'S10M', 'RH2M', 'PMSL',
#       'Q2M', 'CCLOW', 'TD2M', 'GMAX', 'fmisid', 'obs_lat', 'obs_lon',
#       'obs_elevation', 'RH_PT1M_AVG', 'TA_PT1M_AVG', 'WS_PT10M_AVG',
#       'WG_PT10M_MAX', 'Tero', 'RHero', 'WSero', 'WGero', 'ElevD', 'T0bias',
#       'RH0bias', 'WS0bias', 'WG0bias', 'T1bias', 'RH1bias', 'WS1bias',
#       'WG1bias'


# modify the pd frame used in ML training & realtime production
def modify(data,param):
        #data['validdate'] = pd.to_datetime(data['validdate'])
        #data = data.assign(diff_elev=data.model_elevation-data.elevation)
        if param == 'T2m':
                remove = ['index','model_elevation','lat','lon','RH_PT1M_AVG','WS_PT10M_AVG',
                'WG_PT1H_MAX','fcdate','D10M','GMAX','fmisid','RHero','WSero','WGero','RH0bias',
                'WS0bias','WG0bias','RH1bias','WS1bias','WG1bias']
        elif param == 'RH':
                remove = ['index','model_elevation','lat','lon','TA_PT1M_AVG','WS_PT10M_AVG',
                'WG_PT1H_MAX','fcdate','D10M','GMAX','fmisid','Tero','WSero','WGero','T0bias',
                'WS0bias','WG0bias','T1bias','WS1bias','WG1bias']
                #data.drop(data[data['RH_PT10M_AVG'] >= ].index,inplace=True)
        elif param == 'WS':
                remove = ['index','model_elevation','lat','lon','RH_PT1M_AVG','TA_PT1M_AVG',
                'WG_PT1H_MAX','fcdate','RH2M','Q2M','fmisid','RHero','Tero','WGero','RH0bias',
                'T0bias','WG0bias','RH1bias','T1bias','WG1bias']
                data.drop(data[data['WS_PT10M_AVG'] >= 45].index,inplace=True)

        elif param == 'WG':
                remove = ['index','model_elevation','lat','lon','RH_PT1M_AVG','WS_PT10M_AVG',
                'TA_PT1M_AVG','fcdate','Q2M','CCLOW','fmisid','RHero','WSero','Tero','RH0bias',
                'WS0bias','T0bias','RH1bias','WS1bias','T1bias']
                data.drop(data[data['WG_PT1H_MAX'] >= 60].index,inplace=True)

        if 'T_weight' in data.columns:
                data = data.drop('T_weight', axis = 1)
        if 'RH_weight' in data.columns:
                data = data.drop('RH_weight', axis = 1)
        if 'WS_weight' in data.columns:
                data = data-drop('WS_weight', axis = 1)

	# dealing with missing values.
	# this point all rows with Na values are removed
        data = data.drop(remove, axis=1)

        # modify time to sin/cos representation
        data = data.assign(month=data.validdate.dt.month)
        data = data.assign(hour=data.validdate.dt.hour)
        data = encode(data, 'month', 12)
        data = encode(data, 'hour', 24)
        data = data.drop(['month','hour'], axis=1)
        data = data.dropna()
        # reorder the data to be sure that the order is the same in training/prediction
        if param == 'T2m' and 'oldB_T' in data.columns:
                data = data[['leadtime', 'SID', 'validdate', 'T2M', 'S10M', 'RH2M',
                'PMSL', 'Q2M', 'CCLOW', 'obs_lat', 'obs_lon',
                'TA_PT1M_AVG', 'Tero', 'ElevD', 'obs_elevation', 'T0bias','T1bias', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos','oldB_T']]
        elif param == 'T2m' and 'oldB_T' not in data.columns:
                data = data[['leadtime', 'SID', 'validdate', 'T2M', 'S10M', 'RH2M',
                'PMSL', 'Q2M', 'CCLOW', 'obs_lat', 'obs_lon',
                'TA_PT1M_AVG', 'Tero', 'ElevD', 'obs_elevation', 'T0bias','T1bias', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']]
        elif param == 'WS' and 'oldB_WS' in data.columns:
                data = data[['leadtime', 'SID', 'validdate', 'D10M', 'T2M', 'S10M', 
                'PMSL', 'CCLOW', 'GMAX', 'obs_lat', 'obs_lon',
                'WS_PT10M_AVG', 'WSero', 'ElevD', 'obs_elevation', 'WS0bias','WS1bias','month_sin', 'month_cos', 'hour_sin', 'hour_cos','oldB_WS']]
        elif param == 'WS' and 'oldB_WS' not in data.columns:
                data = data[['leadtime', 'SID', 'validdate', 'D10M', 'T2M', 'S10M', 
                'PMSL', 'CCLOW', 'GMAX', 'obs_lat', 'obs_lon',
                'WS_PT10M_AVG', 'WSero', 'ElevD', 'obs_elevation', 'WS0bias','WS1bias', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']]
        elif param == 'RH' and 'oldB_RH' in data.columns:
                data = data[['leadtime', 'SID', 'validdate', 'T2M', 'S10M', 'RH2M',
                'PMSL', 'Q2M', 'CCLOW', 'obs_lat', 'obs_lon',
                'RH_PT1M_AVG', 'RHero', 'ElevD', 'obs_elevation', 'RH0bias','RH1bias', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos','oldB_RH']]
        elif param == 'RH' and 'oldB_RH' not in data.columns:
                data = data[['leadtime', 'SID', 'validdate', 'T2M', 'S10M', 'RH2M',
                'PMSL', 'Q2M', 'CCLOW', 'obs_lat','obs_lon',
		'RH_PT1M_AVG', 'RHero', 'ElevD', 'obs_elevation', 'RH0bias','RH1bias', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']]
        elif param == 'WG':
                data = data[['leadtime', 'SID', 'validdate', 'D10M', 'T2M','S10M', 'RH2M',
                'PMSL', 'GMAX', 'obs_lat', 'obs_lon',
		'WG_PT1H_MAX', 'WGero', 'ElevD', 'obs_elevation', 'WG0bias','WG1bias', 'month_sin', 'month_cos','hour_sin', 'hour_cos']]

	# modify data precision from f64 to f32
        data[data.select_dtypes(np.float64).columns] = data.select_dtypes(np.float64).astype(np.float32)
        return data

def detect_outliers_zscore(data,low_thres,up_thres):
    # remove outliers based on zscore with separate thresholds for upper and lower tail

    outliers = []

    tmpobs = data.iloc[:, 11]
    mean = np.mean(tmpobs)
    std = np.std(tmpobs)
    for i in tmpobs:
        z = (i - mean) / std
        if z > up_thres or z < low_thres:
            outliers.append(i)
    dataout = data[~data.iloc[:, 11].isin(outliers)]
    # print(obs_data[obs_data.iloc[:,5].isin(outliers)])
    return outliers, dataout

def obs_outliers(param,df):

	ajat = sorted(df['validdate'].unique().tolist())
	qcdata = pd.DataFrame()
	outliers = np.empty(0)

	# remove outliers based on zscore with separate thresholds for upper and lower tail
	if param == "RH":
		upper_threshold = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
		lower_threshold = [-4.5,-4.5,-4.5,-4.5,-4.5,-4.5,-4.5,-4.5,-4.5,-4.5,-4.5,-4.5,]
	elif param == "T2m":
		lower_threshold = [-6, -6, -5, -4, -4, -4, -4, -4, -4, -5, -6, -6]
		upper_threshold = [2.5, 2.5, 2.5, 3, 4, 5, 5, 5, 3, 2.5, 2.5, 2.5]
	elif param == "WS" or param == "WG":
		upper_threshold = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
		lower_threshold = [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]

	#xgb2 = [None] * len(ajat)
	for i in range(0,len(ajat)):
		#print(ajat[i])
		tmp = df[df['validdate']==ajat[i]]
		tmpobs = tmp.iloc[:,11] #.values #RHero --> mallivirhe
		#print(tmpobs)

		thres_month = ajat[i].month
		up_thres = upper_threshold[thres_month - 1]
		low_thres = lower_threshold[thres_month - 1]
		pois, mukaan = detect_outliers_zscore(tmp, low_thres,up_thres)
		outliers = np.append(outliers,pois)
		qcdata = qcdata.append(mukaan, ignore_index=False)
	#print(outti)
	print("out pituus:",len(outliers))
	if len(outliers) > 0:
		print('isoin:',np.max(outliers))
		print('pienin:',np.min(outliers))
	plt.figure(3)
	orig1 = df.iloc[:,11]
	orig1 = orig1[~np.isnan(orig1)]
	all1 = [orig1, outliers, qcdata.iloc[:,11]]
	plt.boxplot(all1)
	plt.title(" outliers lkm " + str(len(outliers)) )
	plt.savefig(param + '_Zscore.png')
	return qcdata
