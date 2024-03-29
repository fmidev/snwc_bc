## Model development/training

#### Data

* * *

The MetCoOp nowcast (MNWC) data used for training is downloaded from MetCoOp Arcus archive (https://metcoop.smhi.se/dokuwiki/nwp/metcoop/arcus/userguide) where MNWC data is stored for SYNOP point locations in vfld format. The SYNOP observations are downloaded from FMI Climate database using Smartmet server. Pandas dataframes are used for ML training, but data is stored in feather format (.ftr) in s3://lake.fmi.fi/ml-data starting from year 2020. Each .ftr-file contains 6 months of data "q12" referring to months Jan-Jun and "q34" to Jul-Dec. Depending on the parameter forecasted, the input features used by the model vary slightly.

| feature | explanation | unit |
| --- | --- | --- |
| leadtime | MNWC forecast leadtime | hour |
| T2M | MNWC 2m temperature | C   |
| D10M | MNWC 10m wind direction | Degrees |
| S10M | MNWC 10m wind speed | m/s |
| RH2M | MNWC relative humidity | % \[0-100\] |
| PMSL | MNWC mean sea level pressure | hPa |
| Q2M | MNWC specific humidity | kg/kg |
| CCLOW | MNWC low level cloud cover | \[0-1\] |
| obs\_lat | observation latitude |     |
| obs\_lon | observation longitude |     |
| obs\_elevation | Height of the observation station | m   |
| 1bias | T2m,WS,WG,RH latest forecast-obs |     |
| ero | T2m,WS,WG,RH forecast-obs error (y/Truth) |     |
| month\_sin/month\_cos | cyclical month |     |
| hour\_sin/hour\_cos | cyclical hour of the day |     |
| ElevD | Elevation difference of model-observation | m   |

#### Data preprocessing

* * *

MNWC data can be downloaded from Arcus archive in vfld format using s3cmd. In HARP package there is an R function for converting vfld files to sqlite tables (\[[https://rdrr.io/github/andrew-MET/harpIO/man/read\\\_forecast.html](https://rdrr.io/github/andrew-MET/harpIO/man/read%5C_forecast.html)\]) These several sqlite tables generated by the code are further merged to one. To keep the data amounts reasonable, only one month of data is downloaded at a time. These unprocessed MNWC data sqlite tables from Arcus are also stored to s3 /ml-data bucket.

The data is further preprocessed by adding observation information from FMI's Climate database and by combining monthly data to 6 month datasets in feather (.ftr) format which can be fed to the machine learning model for training.

#### Training

* * *

Training the gradient boosting regressor model is done using similar setup for all the parameters forecasted: T2m, WS, WG and RH, only the input features used in training vary slightly. Currently random 10% subset of the training data is used for validation.

#### Files

* * *

- **Dockerfile** Includes description on the Docker image used at Openshift
- **requirements.txt** Python dependencies for container
- **README.md** File containing documentation 
- **biasc.py** Code to produce the realtime bias correction
- **create\_training\_data.sh** \[starttime endtime\] Script to 1) download MNWC data from arcus 2) convert it to sqlite (**read\_vfld.R**) 3) merge sqlite files (**merge\_HARP\_files.py**). Example of use: sh create\_training\_data.sh yyyymmddhhss yyyymmddhhss
- **preprocess.py** \[ --starttime --months --path --output\] Code to add observations to MNWC data and to convert it to feather format with several months of data. Example of use: python3 preprocess.py --starttime 202301 --months 6 --path '/MNWC\_data/monthly/sqlite/' --output 'mnwc2023q12.ftr'
- **run\_xgb\_training.sh** Script to run model training for different parameter with **xgb\_train\_model.py** the hyperparameters are hard coded for different parameters. Input data are the feather files produced by preprocess.py that are further modified by **XGBmodify.py** to pandas dataframe.
- **xgb\_test\_results.py** \[--param --path (to test data) --model1 --model2\] Code to compare basic RMSE over leadtime results with previous ML model version

&nbsp;

&nbsp;
