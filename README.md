# Machine learning based bias correction for short range deterministic weather forecasts
This bias correction (BC) can be used to correct the forecast errors in deterministic weather model forecasts up to 0-12 hour leadtimes. The input data is the model grids (grib2-files) and the output is bias corrected model fields (grib2) for forecast hours 0-input leadtimes. Smartmet server (only available at FMI) is used to get observation information needed in the real-time production. Bias correction is a gradien boosting regressor based on 2 years of training data and is calculated for points which are then further gridded back to NWP model background using Gridpp. Bias correction is available for following parameters: 2m temperature, 10m wind speed, 10m wind gust and 2m relative humidity.  

## Usage 
```
biasc.py --topography_data mnwc-Z-M2S2.grib2 --landseacover mnwc-LC-0TO1.grib2 --t2_data mnwc-T-K.grib2 --wg_data mnwc-FFG-MS.grib2 --nl_data mnwc-NL-0TO1.grib2 --ppa_data mnwc-P-PA.grib2 --wd_data mnwc-DD-D.grib2 --q2_data mnwc-Q-KGKG.grib2 --ws_data mnwc-FF-MS.grib2 --rh_data mnwc-RH-0TO1.grib2 --output T-K.grib2 --parameter temperature
```
* Run time of ~20min past recommended to include sufficient amount of observations  

## Authors
leila.hieta@fmi.fi mikko.partio@fmi.fi

## Known issues, features and development ideas
* Machine learning model files are not included in this repository
* Keyword "snwc" used to fetch obs data from smartmet server (constant list of SYNOP stations)   
* SYNOP observations used for error correction, for temperature also NetAtmo is used   
* NetAtmo station altitudes are interpolated from digital elevation map (not included to this repository) 
 
