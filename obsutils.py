import numpy as np
import pandas as pd
from flatten_json import flatten
import requests
import datetime
import gridpp
import os
import rioxarray
import pyproj
from scipy.interpolate import RegularGridInterpolator

def read_conventional_obs(args, fcstime, mnwc, analysistime):
    parameter = args.parameter
    # read observations for "analysis time" == leadtime 1
    obstime = fcstime[1]
    # print("Observations are from time:", obstime)

    timestr = obstime.strftime("%Y%m%d%H%M%S")
    trad_obs = []

    # define obs parameter names used in observation database, for ws the potential ws values are used for Finland
    if parameter == "temperature":
        obs_parameter = "TA_PT1M_AVG"
    elif parameter == "windspeed":
        obs_parameter = (
            "WSP_PT10M_AVG"  # potential wind speed available for Finnish stations
        )
    elif parameter == "gust":
        obs_parameter = "WG_PT1H_MAX"
    elif parameter == "humidity":
        obs_parameter = "RH_PT1M_AVG"

    # conventional obs are read from two distinct smartmet server producers
    # if read fails, abort program

    for producer in ["observations_fmi", "foreign"]:
        if producer == "foreign" and parameter == "windspeed":
            obs_parameter = "WS_PT10M_AVG"
        url = "http://smartmet.fmi.fi/timeseries?producer={}&tz=gmt&precision=auto&starttime={}&endtime={}&param=fmisid,longitude,latitude,utctime,elevation,{}&format=json&keyword=snwc".format(
            producer, timestr, timestr, obs_parameter
        )

        resp = requests.get(url)

        testitmp = []
        testitmp2 = []
        if resp.status_code == 200:
            testitmp2 = pd.DataFrame(resp.json())
            # test if all the retrieved observations are Nan
            testitmp2 = testitmp2[obs_parameter].isnull().all()

        if resp.status_code != 200 or testitmp2 == True or resp.json == testitmp:
            print(
                "Not able to connect Smartmet server for observations, original MNWC fields are saved"
            )
            # Remove analysistime (leadtime=0), because correction is not made for that time
            fcstime.pop(0)
            mnwc = mnwc[1:]
            if parameter == "humidity":
                mnwc = mnwc / 100
            elif parameter == "temperature":
                mnwc = mnwc + 273.15
            write_grib(args, analysistime, fcstime, mnwc)
            sys.exit(1)
        trad_obs += resp.json()

    obs = pd.DataFrame(trad_obs)
    # rename observation column if WS, otherwise WS and WSP won't work
    if parameter == "windspeed":  # merge columns for WSP and WS
        obs["WSP_PT10M_AVG"] = obs["WSP_PT10M_AVG"].fillna(obs["WS_PT10M_AVG"])

    obs = obs.rename(columns={"fmisid": "station_id"})

    count = len(trad_obs)
    print("Got {} traditional obs stations for time {}".format(count, obstime))

    if count == 0:
        print(
            "Number of observations from Smartmet serve is 0, original MNWC fields are saved"
        )
        fcstime.pop(0)
        mnwc = mnwc[1:]
        if parameter == "humidity":
            mnwc = mnwc / 100
        elif parameter == "temperature":
            mnwc = mnwc + 273.15
        write_grib(args, analysistime, fcstime, mnwc)
        sys.exit(1)

    # print(obs.head(5))
    print("min obs:", min(obs.iloc[:, 5]))
    print("max obs:", max(obs.iloc[:, 5]))
    return obs


def read_netatmo_obs(args, fcstime):
    # read Tiuha db NetAtmo observations for "analysis time" == leadtime 1
    snwc1_key = os.environ.get("SNWC1_KEY")
    assert snwc1_key is not None, "tiuha api key not find (env variable 'SNWC1_KEY')"

    os.environ["NO_PROXY"] = "tiuha-dev.apps.ock.fmi.fi"
    obstime = fcstime[1]

    url = "https://tiuha-dev.apps.ock.fmi.fi/v1/edr/collections/netatmo-air_temperature/cube?bbox=4,54,32,71.5&start={}Z&end={}Z".format(
        (obstime - datetime.timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S"),
        obstime.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    headers = {"Authorization": f"Basic {snwc1_key}"}
    # os.environ['NO_PROXY'] = 'tiuha-dev.apps.ock.fmi.fi'
    resp = requests.get(url, headers=headers)

    crowd_obs = None
    testitmp = []
    testitmp2 = []

    if resp.status_code == 200:
        testitmp2 = pd.DataFrame(resp.json())

    if resp.status_code != 200 or resp.json() == testitmp or len(testitmp2) == 0:
        print("Error fetching NetAtmo data, status code: {}".format(resp.status_code))
    else:
        crowd_obs = resp.json()
        # print("Got {} crowd sourced obs stations".format(len(crowd_obs)))

    obs = None

    if crowd_obs is not None:
        flattened_data = [flatten(feature) for feature in crowd_obs["features"]]
        obs = pd.DataFrame(flattened_data)
        obs.drop(obs.columns[[0, 1, 5, 6, 7, 9, 10, 11, 12]], axis=1, inplace=True)
        obs = obs.rename(
            columns={
                "geometry_coordinates_0": "longitude",
                "geometry_coordinates_1": "latitude",
                "geometry_coordinates_2": "station_id",
                "properties_resultTime": "utctime",
                "properties_result": "temperature",
            }
        )
        # Remove duplicated observations/station by removing duplicated lat/lon values and keep the first value only
        obs = obs[~obs.duplicated(subset=["latitude", "longitude"], keep="first")]
        print("Got {} crowd sourced obs stations".format(len(obs)))

        # netatmo obs do not contain elevation information, but we need thatn
        # to have the best possible result from optimal interpolation
        #
        # use digital elevation map data to interpolate elevation information
        # to all netatmo station points

        # print("Interpolating elevation to NetAtmo stations")
        dem = rioxarray.open_rasterio(args.dem_data)

        # dem is projected to lambert, our obs data is in latlon
        # transform latlons to projected coordinates

        ll_to_proj = pyproj.Transformer.from_crs("epsg:4326", dem.rio.crs)
        xs, ys = ll_to_proj.transform(obs["latitude"], obs["longitude"])
        obs["x"] = xs
        obs["y"] = ys

        # interpolated dem data to netatmo station points in x,y coordinates

        demds = dem.to_dataset("band").rename({1: "dem"})
        x = demds["x"].values

        # RegularGridInterpolator requires y axis value to be ascending -
        # geotiff is always descending

        y = np.flip(demds["y"].values)
        z = np.flipud(demds["dem"].values)

        interp = RegularGridInterpolator(points=(y, x), values=z)

        points = np.column_stack((obs["y"], obs["x"]))
        obs["elevation"] = interp(points)

        obs = obs.drop(columns=["x", "y"])
        # print(obs.head(5))
        # reorder/rename columns of the NetAtmo df to match with synop data
        obs = obs[
            [
                "station_id",
                "longitude",
                "latitude",
                "utctime",
                "elevation",
                "temperature",
            ]
        ]
        obs.rename(columns={"temperature": "TA_PT1M_AVG"}, inplace=True)
        # print(obs.head(10))
        print("min NetAtmo obs:", min(obs.iloc[:, 5]))
        print("max NetAtmo obs:", max(obs.iloc[:, 5]))

    return obs


def detect_outliers_zscore(args, fcstime, obs_data):
    # remove outliers based on zscore with separate thresholds for upper and lower tail
    if args.parameter == "humidity":
        upper_threshold = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        lower_threshold = [-4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5]
    elif args.parameter == "temperature":
        lower_threshold = [-6, -6, -5, -4, -4, -4, -4, -4, -4, -5, -6, -6]
        upper_threshold = [2.5, 2.5, 2.5, 3, 4, 5, 5, 5, 3, 2.5, 2.5, 2.5]
    elif args.parameter == "windspeed" or args.parameter == "gust":
        upper_threshold = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        lower_threshold = [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]

    thres_month = fcstime.month
    up_thres = upper_threshold[thres_month - 1]
    low_thres = lower_threshold[thres_month - 1]

    outliers = []

    tmpobs = obs_data.iloc[:, 5]
    mean = np.mean(tmpobs)
    std = np.std(tmpobs)
    for i in tmpobs:
        z = (i - mean) / std
        if z > up_thres or z < low_thres:
            outliers.append(i)
    dataout = obs_data[~obs_data.iloc[:, 5].isin(outliers)]
    # print(obs_data[obs_data.iloc[:,5].isin(outliers)])
    return outliers, dataout


def read_obs(args, fcstime, grid, lc, mnwc, analysistime):
    """Read observations from smartmet server"""

    # read observations for "analysis" time == leadtime 1
    # obstime = fcstime[1]

    obs = read_conventional_obs(args, fcstime, mnwc, analysistime)

    # for temperature we also have netatmo stations
    # these are optional

    if args.parameter == "temperature":
        netatmo = read_netatmo_obs(args, fcstime)
        if netatmo is not None:
            obs = pd.concat((obs, netatmo))

        # obs["temperature"] += 273.15

    outliers, obs = detect_outliers_zscore(args, fcstime[1], obs)
    print("removed " + str(len(outliers)) + " outliers from observations")
    # print(outliers)

    print("min of QC obs:", min(obs.iloc[:, 5]))
    print("max of QC obs:", max(obs.iloc[:, 5]))

    points1 = gridpp.Points(
        obs["latitude"].to_numpy(),
        obs["longitude"].to_numpy(),
    )
    # interpolate nearest land sea mask values from grid to obs points (NWP data used, since there's no lsm info from obs stations available)
    obs["lsm"] = gridpp.nearest(grid, points1, lc)

    points = gridpp.Points(
        obs["latitude"].to_numpy(),
        obs["longitude"].to_numpy(),
        obs["elevation"].to_numpy(),
        obs["lsm"].to_numpy(),
    )

    return points, obs
