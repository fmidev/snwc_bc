#!/bin/bash
# Acript to run XGB training code in cron

# directory in devmos
cd /data/hietal/VFLD
export TMPDIR=/data/hietal/tmp_temp

#python3.9 xgb_train_model.py --param 'T2m' --path '/data/hietal/XGB/'
#python3.9 xgb_train_model.py --param 'WS' --path '/data/hietal/XGB/'
python3.9 xgb_train_model.py --param 'WG' --path '/data/hietal/XGB/'
#python3.9 xgb_train_model.py --param 'RH' --path '/data/hietal/XGB/'
