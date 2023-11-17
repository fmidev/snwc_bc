#!/bin/bash
# Script to create training data for MNWC based nowcast bias correction
# 1) Download MNWC data from Arcus
# 2) Convert data from vfld files to sqlite using R (HARP package https://rdrr.io/github/andrew-MET/harpIO/man/read_forecast.html)
# 3) merge several SQLite files to one

# PARAMETER INPUTS: $1 is the starttime (yyyymmddhhss), $2 is endtime

# modify this!
base="/data/hietal/VFLD"

starttime=$1
endtime=$2

echo $starttime
echo $endtime
year=$(echo $starttime | cut -c1-4)
month=$(echo $starttime | cut -c5-6)

# R function assumes that the vfld files are in specific directory tree
folder_name="MNWC$year/MNWC_preop"

if [ ! -d "$folder_name" ]; then
    mkdir -p "$folder_name"
    echo "Folder created successfully!"
else
    echo "Folder already exists."
fi

cd $folder_name
#mkdir -p MNWC$year/MNWC_preop/ && cd $_
pwd

# download data from Arcus usin S3
s3cmd -c ~/.s3cfg-arcus --host=arcus-s3.nsc.liu.se --host-bucket='' --recursive get s3://verif/vfld/MNWC_preop/$year/$month/

# move files up one directory
find . -mindepth 2 -type f -print -exec mv {} . \;
# remove empty directories
find -empty -type d -delete
# unzip the files
find -name "*tar.gz" -exec tar xvzf '{}' \;
# remove the tar files
rm *.tar.gz
# since 2023 the name of the MNWC files was changed in Arcus, we need to rename the files for the R code to work
rename "preop0" "preop" *

# move back to the main directory
cd ../../

# make relevant directories
folder_name1="tmp"
folder_name2="output"

if [ ! -d "$folder_name1" ]; then
    mkdir -p "$folder_name1"
else
    echo "Folder already exists."
fi
if [ ! -d "$folder_name2" ]; then
    mkdir -p "$folder_name2"
else
    echo "Folder already exists."
fi

# R code: vfld --> sqlite
# R code converts files to multiple sqlite-files in tmp directory
Rscript read_vfld.R $starttime $endtime $base

# python code to merge sqlite files
python3 merge_HARP_files.py --path $base/tmp/MNWC_preop/$year/$month/ --outfile $base/output/MNWC$year$month.sqlite

# remove temp sqlite files from /tmp
rm -r tmp/*
rm -r MNWC$year/*

