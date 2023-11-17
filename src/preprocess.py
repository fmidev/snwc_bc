import pandas as pd
import sqlite3
import datetime
from time import gmtime, strftime
import time
import requests
import io
import feather
import re
import sys,os,getopt
# read obs from Smartmet server
def readobs_all(starttime,endtime,t_reso,producer,obs_param):
    url = 'http://smartmet.fmi.fi/timeseries'
    params = "wmo,fmisid,latitude,longitude,time,elevation,"+obs_param

    payload1 = {
       "keyword": "snwc",
       "tz":"utc",
       "separator":",",
       "producer": producer,
       "precision": "double",
       "timeformat": "sql",
       "param": params,
       "starttime": starttime,   #"2020-09-18T00:00:00",
       "endtime": endtime,   #"2020-09-18T00:00:00",
       "timestep": t_reso,
       "format": "ascii"
    }
    r = requests.get(url, params=payload1)
    return r

def merge_data(path,kk): #alku,loppu):
   print(path)
   print(kk)
   start_time = time.time() # laske suoritusaika
   #aikasql = pd.to_datetime(alku).strftime("%Y%m")

   # Read sqlite query results into a pandas DataFrame
   con = sqlite3.connect(path+kk)
   #df = pd.read_sql_query("SELECT DISTINCT SID FROM FC",con)
   #df = pd.read_sql_query("SELECT leadtime,parameter,SID,lat,lon,model_elevation,validdate,MNWC_preop_det FROM FC limit 2000000", con)

   df = pd.read_sql_query("SELECT leadtime, SID, lat, lon, model_elevation, validdate, fcdate, \
      avg(case when PARAMETER = 'D10m' THEN MNWC_preop_det END) AS D10M, \
      avg(case when PARAMETER = 'T2m' THEN MNWC_preop_det END) AS T2M, \
      avg(case when PARAMETER = 'S10m'  THEN MNWC_preop_det END) AS S10M, \
      avg(case when PARAMETER = 'RH2m'  THEN MNWC_preop_det END) AS RH2M, \
      avg(case when PARAMETER = 'Pmsl'  THEN MNWC_preop_det END) AS PMSL, \
         avg(case when PARAMETER = 'Q2m'  THEN MNWC_preop_det END) AS Q2M, \
         avg(case when PARAMETER = 'CClow'  THEN MNWC_preop_det END) AS CCLOW, \
         avg(case when PARAMETER = 'Gmax'  THEN MNWC_preop_det END) AS GMAX, \
         COUNT(*) AS rows_n FROM FC WHERE parameter IN ('T2m','D10m','S10m','RH2m','Pmsl','Q2m','CClow','Gmax') \
         GROUP BY leadtime, SID, LAT, LON, MODEL_ELEVATION,VALIDDATE, FCDATE", con)
   con.close()
   print(df['validdate'][1:5])
   obstime = pd.to_datetime(df['validdate'], unit='s') # aika havaintohakua varten
   df['validdate'] = pd.to_datetime(df['validdate'], origin='unix', unit='s',utc=True)
   print(df['validdate'][1:5])
   #exit()
   #datetime.datetime.utcfromtimestamp(df['validdate']) #pd.to_datetime(df['validdate'], unit='s', utc=True).dt.tz_convert('Europe/Helsinki')
   #print(df['validdate'])
   df['fcdate'] = pd.to_datetime(df['fcdate'], origin='unix', unit='s', utc=True)

   # define times for obs retrieval
   alku = min(obstime) #df['validdate'].min()
   loppu = max(obstime) #df['validdate']. max()
   print(alku)
   print(loppu)

   # runtime info
   print("%s seconds" % (time.time() - start_time))
   obs_param = 'RH_PT1M_AVG,TA_PT1M_AVG,WSP_PT10M_AVG,WG_PT1H_MAX'  # WG_10M_MAX muutettu 1h!, WSP_PT10M_AVG
   AA = readobs_all(alku,loppu,'1h','observations_fmi',obs_param).content
   obs_param = 'RH_PT1M_AVG,TA_PT1M_AVG,WS_PT10M_AVG,WG_PT1H_MAX'
   BB = readobs_all(alku,loppu,'1h','foreign',obs_param).content

   colnames=['SID','fmisid','obs_lat','obs_lon','validdate','obs_elevation','RH_PT1M_AVG','TA_PT1M_AVG','WS_PT10M_AVG','WG_PT1H_MAX']
   AAA = pd.read_csv(io.StringIO(AA.decode('utf-8')),names=colnames,header=None )
   BBB = pd.read_csv(io.StringIO(BB.decode('utf-8')),names=colnames,header=None )
   OBS = pd.concat([AAA, BBB])
   print(OBS.head(10))
   #print(OBS.dtypes)
   OBS['validdate'] = pd.to_datetime(OBS['validdate'], utc=True)
   print(OBS['validdate'].min())
   print(OBS['validdate'].max())

   print("%s seconds" % (time.time() - start_time))

   #print(OBS.head(20))
   #print(df.head(20))

   OBS['SID'].astype('int64')

   df_new = pd.merge(df, OBS,  how='left', left_on=['SID','validdate'], right_on = ['SID','validdate'])

   # calculate F-O errors
   df_new['T2M'] = df_new['T2M'] - 273.15
   #df_new['TD2M'] = df_new['TD2M'] - 273.15
   df_new['Tero'] = df_new['T2M'] - df_new['TA_PT1M_AVG']
   df_new['RHero'] = df_new['RH2M'] - df_new['RH_PT1M_AVG']
   df_new['WSero'] = df_new['S10M'] - df_new['WS_PT10M_AVG']
   df_new['WGero'] = df_new['GMAX'] - df_new['WG_PT1H_MAX']
   # elevation difference
   df_new['ElevD'] = df_new['model_elevation'] - df_new['obs_elevation']

   ####Erottele analyysiajan 0h ja 1h virheet ja yhdistä
   ajat = sorted(df_new['leadtime'].unique().tolist())
   lt0 = df_new[df_new['leadtime']==0]
   lt0 = lt0[['fcdate','SID','Tero','RHero','WSero','WGero']]
   lt0.columns = ['fcdate', 'SID','T0bias','RH0bias','WS0bias','WG0bias']
   lt1 = df_new[df_new['leadtime']==1]
   lt1 = lt1[['fcdate','SID','Tero','RHero','WSero','WGero']]
   lt1.columns = ['fcdate', 'SID','T1bias','RH1bias','WS1bias','WG1bias']
   df_new[(df_new.leadtime != 0)] # & (df_new.ladtime != 1)]

   df_new = pd.merge(df_new, lt0,  how='left', left_on=['SID','fcdate'], right_on = ['SID','fcdate'])
   df_new = pd.merge(df_new, lt1,  how='left', left_on=['SID','fcdate'], right_on = ['SID','fcdate'])

   #tiputa tieto sqlite haettavien lkm
   df_new = df_new.drop(['rows_n'],axis=1)
   # tiputa rivit, joilta kaikki havainnot puuttuu (esim ei toiminnassa olevat asemat)
   df_new = df_new.dropna( how='all', subset=['RH_PT1M_AVG','TA_PT1M_AVG','WS_PT10M_AVG','WG_PT1H_MAX'])

   return df_new


def main():

   options, remainder = getopt.getopt(sys.argv[1:],[],['starttime=','months=','path=','output=','help'])

   for opt, arg in options:
       if opt == '--help':
           print('preprocess.py starttime=[yyyymm] months=how many months of data to merge path=directory for the input sqlite files ends with /, output=path & name of the output[utcmnwc2023q12.ftr]')
           exit()
       elif opt == '--starttime':
               starttime = arg
       elif opt == '--months':
               months = arg
       elif opt == '--path':
               path = arg
       elif opt == '--output':
               output = arg

   try:
       starttime, months, path, output
   except NameError:
       print('ERROR! Not all input parameters specified: ')
       exit()

   # list all sqlite files (per month)
   listf = os.listdir(path)

   print("kk numerona",int(months))

   # chronological ordering
   def get_date(filename):
      #print(filename.split('_')[1][:6])
      return re.findall(r'\d+', filename)

   sortfils = sorted(listf, key=get_date)
   print(sortfils)

   # get list index of the starttime (yyyymm) file
   def get_file_index_by_number(files, number):
      for i, file in enumerate(files):
         if number in file:
            return i
      return -1

   first_ind = get_file_index_by_number(sortfils, str(starttime))
   print(first_ind)
   print(sortfils[first_ind])

   # merge data for the first month (starttime1)
   data = merge_data(path, sortfils[first_ind])
   # read the following months
   for x in range(1,int(months)):
        print(x)
        print(sortfils[first_ind+x])
        tmp_data = merge_data(path, sortfils[first_ind+x])
        data = pd.concat([data,tmp_data])

   data = data.reset_index()
   data.to_feather(output)


if __name__ == "__main__":
    main()



