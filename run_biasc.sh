#!/usr/bin/python3.9
# Script to run ML realtime forecasts for testing

#python3.9 --version

#cd /home/users/hietal/statcal/python_projects/snwc_bc/

parameter=$1 # T2m, RH, WS or WG

WEEKDAY=`date +"%a"`
HOD=`date +"%H"`
AIKA1=`date "+%Y%m%d%H"  -u`
HH=2
NN=$(($AIKA1-$HH))
bucket="s3://routines-data/mnwc-biascorrection/production/"
echo $NN

#export TMPDIR=/data/hietal/testi

if [ "$parameter" == "T2m" ]; then
  pyparam="temperature"
elif [ "$parameter" == "RH" ]; then
  pyparam="humidity"
elif [ "$parameter" == "WS" ]; then
  pyparam="windspeed"
elif [ "$parameter" == "WG" ]; then
  pyparam="gust"
else
  echo "parameter must be T2m, RH, WS or WG"
  exit 1
fi

python3.9 biasc.py --topography_data "$bucket""$NN"00/Z-M2S2.grib2 --landseacover "$bucket""$NN"00/LC-0TO1.grib2 --t2_data "$bucket""$NN"00/T-K.grib2 --wg_data "$bucket""$NN"00/FFG-MS.grib2 --nl_data "$bucket""$NN"00/NL-0TO1.grib2 --ppa_data "$bucket""$NN"00/P-PA.grib2 --wd_data "$bucket""$NN"00/DD-D.grib2 --q2_data "$bucket""$NN"00/Q-KGKG.grib2 --ws_data "$bucket""$NN"00/FF-MS.grib2 --rh_data "$bucket""$NN"00/RH-0TO1.grib2 --output testi_"$parameter".grib2 --parameter "$pyparam"

#rm -r /data/hietal/testi/tmp*
