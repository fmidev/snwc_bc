import gridpp
import numpy as np
import eccodes as ecc
import sys
import pyproj
import requests
import datetime
import argparse
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import fsspec
import os
import time
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import copy
import numpy.ma as ma
import warnings
from multiprocessing import Process, Queue
warnings.filterwarnings('ignore')

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topography_data", action="store", type=str, required=True)
    parser.add_argument("--landseacover_data", action="store", type=str, required=True)
    parser.add_argument("--parameter", action="store", type=str, required=True)
    parser.add_argument("--wg_data", action="store", type=str, required=True)
    parser.add_argument("--ppa_data", action="store", type=str, required=True)
    parser.add_argument("--ws_data", action="store", type=str, required=True)
    parser.add_argument("--rh_data", action="store", type=str, required=True)
    parser.add_argument("--t2_data", action="store", type=str, required=True)
    parser.add_argument("--wd_data", action="store", type=str, required=True)
    parser.add_argument("--q2_data", action="store", type=str, required=True)
    parser.add_argument("--nl_data", action="store", type=str, required=True)
    parser.add_argument(
        "--dem_data", action="store", type=str, default="DEM_100m-Int16.tif"
    )
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--disable_multiprocessing", action="store_true", default=False)

    args = parser.parse_args()

    allowed_params = ["temperature", "humidity", "windspeed", "gust"]
    if args.parameter not in allowed_params:

        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)

    return args


def get_shapeofearth(gh):
    """Return correct shape of earth sphere / ellipsoid in proj string format.
    Source data is grib2 definition.
    """

    shape = ecc.codes_get_long(gh, "shapeOfTheEarth")

    if shape == 1:
        v = ecc.codes_get_long(gh, "scaledValueOfRadiusOfSphericalEarth")
        s = ecc.codes_get_long(gh, "scaleFactorOfRadiusOfSphericalEarth")
        return "+R={}".format(v * pow(10, s))

    if shape == 5:
        return "+ellps=WGS84"


def get_falsings(projstr, lon0, lat0):
    """Get east and north falsing for projected grib data"""

    ll_to_projected = pyproj.Transformer.from_crs("epsg:4326", projstr)
    return ll_to_projected.transform(lat0, lon0)


def get_projstr(gh):
    """Create proj4 type projection string from grib metadata" """

    projstr = None

    proj = ecc.codes_get_string(gh, "gridType")
    first_lat = ecc.codes_get_double(gh, "latitudeOfFirstGridPointInDegrees")
    first_lon = ecc.codes_get_double(gh, "longitudeOfFirstGridPointInDegrees")

    if proj == "polar_stereographic":
        projstr = "+proj=stere +lat_0=90 +lat_ts={} +lon_0={} {} +no_defs".format(
            ecc.codes_get_double(gh, "LaDInDegrees"),
            ecc.codes_get_double(gh, "orientationOfTheGridInDegrees"),
            get_shapeofearth(gh),
        )
        fe, fn = get_falsings(projstr, first_lon, first_lat)
        projstr += " +x_0={} +y_0={}".format(-fe, -fn)

    elif proj == "lambert":
        projstr = (
            "+proj=lcc +lat_0={} +lat_1={} +lat_2={} +lon_0={} {} +no_defs".format(
                ecc.codes_get_double(gh, "Latin1InDegrees"),
                ecc.codes_get_double(gh, "Latin1InDegrees"),
                ecc.codes_get_double(gh, "Latin2InDegrees"),
                ecc.codes_get_double(gh, "LoVInDegrees"),
                get_shapeofearth(gh),
            )
        )
        fe, fn = get_falsings(projstr, first_lon, first_lat)
        projstr += " +x_0={} +y_0={}".format(-fe, -fn)

    else:
        print("Unsupported projection: {}".format(proj))
        sys.exit(1)

    return projstr


def read_file_from_s3(grib_file):
    uri = "simplecache::{}".format(grib_file)

    return fsspec.open_local(
        uri, s3={"anon": True, "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"}}
    )


def read_grib(gribfile, read_coordinates=False):
    """Read first message from grib file and return content.
    List of coordinates is only returned on request, as it's quite
    slow to generate.
    """
    forecasttime = []
    values = []

    print(f"Reading file {gribfile}")
    wrk_gribfile = gribfile

    if gribfile.startswith("s3://"):
        wrk_gribfile = read_file_from_s3(gribfile)

    lons=[]
    lats=[]

    with open(wrk_gribfile) as fp:
        # print("Reading {}".format(gribfile))

        while True:
            gh = ecc.codes_grib_new_from_file(fp)

            if gh is None:
                break

            ni = ecc.codes_get_long(gh, "Nx")
            nj = ecc.codes_get_long(gh, "Ny")
            dataDate = ecc.codes_get_long(gh, "dataDate")
            dataTime = ecc.codes_get_long(gh, "dataTime")
            forecastTime = ecc.codes_get_long(gh, "forecastTime")
            analysistime = datetime.datetime.strptime(
                "{}.{:04d}".format(dataDate, dataTime), "%Y%m%d.%H%M"
            )

            ftime = analysistime + datetime.timedelta(hours=forecastTime)
            forecasttime.append(ftime)

            tempvals = ecc.codes_get_values(gh).reshape(nj, ni)
            values.append(tempvals)

            if read_coordinates and len(lons) == 0:
                projstr = get_projstr(gh)

                di = ecc.codes_get_double(gh, "DxInMetres")
                dj = ecc.codes_get_double(gh, "DyInMetres")

                proj_to_ll = pyproj.Transformer.from_crs(projstr, "epsg:4326")

                for j in range(nj):
                    y = j * dj
                    for i in range(ni):
                        x = i * di

                        lat, lon = proj_to_ll.transform(x, y)
                        lons.append(lon)
                        lats.append(lat)


        if read_coordinates == False and len(values) == 1:
            return None, None, np.asarray(values).reshape(nj,ni), analysistime, forecasttime
        elif read_coordinates == False and len(values) > 1:
            return None, None, np.asarray(values), analysistime, forecasttime
        else:
            return (
	np.asarray(lons).reshape(nj, ni),
        np.asarray(lats).reshape(nj, ni),
        np.asarray(values),
        analysistime,
        forecasttime,
    )


def read_grid(args):
    """Top function to read "all" gridded data"""
    # Define the grib-file used as background/"parameter_data"
    if args.parameter == "temperature":
        parameter_data = args.t2_data
    elif args.parameter == "windspeed":
        parameter_data = args.ws_data
    elif args.parameter == "gust":
        parameter_data = args.wg_data
    elif args.parameter == "humidity":
        parameter_data = args.rh_data

    lons, lats, vals, analysistime, forecasttime = read_grib(parameter_data, True)

    _, _, topo, _, _ = read_grib(args.topography_data, False)
    _, _, lc, _, _ = read_grib(args.landseacover_data, False)

    # modify  geopotential to height and use just the first grib message, since the topo & lc fields are static
    topo = topo/9.81
    topo = topo[0]
    lc = lc[0]

    if args.parameter == "temperature":
        vals = vals - 273.15
    elif args.parameter == "humidity":
        vals = vals*100

    grid = gridpp.Grid(lats, lons, topo, lc)
    return grid, lons, lats, vals, analysistime, forecasttime, lc, topo

def read_ml_grid(args):

    _, _, ws, _, _ = read_grib(args.ws_data, False)
    _, _, rh, _, _ = read_grib(args.rh_data, False)
    _, _, q2, _, _ = read_grib(args.q2_data, False)
    _, _, cl, _, _ = read_grib(args.nl_data, False)
    _, _, ps, _, _ = read_grib(args.ppa_data, False)
    _, _, t2, _, _ = read_grib(args.t2_data, False)
    _, _, wg, _, _ = read_grib(args.wg_data, False)
    _, _, wd, _, _ = read_grib(args.wd_data, False)

    # change parameter units:
    rh = rh * 100
    t2 = t2 -273.15
    ps = ps/100
    cl = np.around(cl/0.125,0)

    return ws, rh, t2, wg, cl, ps, wd, q2

def read_conventional_obs(args, fcstime, mnwc):
    parameter = args.parameter
    # read observations for "analysis time" == leadtime 1
    obstime = fcstime[1]
    #print("Observations are from time:", obstime)

    timestr = obstime.strftime("%Y%m%d%H%M%S")
    trad_obs = []

    # define obs parameter names used in observation database, for ws the potential ws values are used for Finland 
    if parameter == "temperature":
        obs_parameter = "TA_PT1M_AVG"
    elif parameter == "windspeed":
        obs_parameter = "WSP_PT10M_AVG" # potential wind speed available for Finnish stations
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

        if resp.status_code != 200:
            print("Not able to connect Smartmet server for observations, original MNWC fields are saved")
            write_grib(args, analysistime, fcstime, mnwc)
            sys.exit(1)
        trad_obs += resp.json()


    obs = pd.DataFrame(trad_obs)
    # rename observation column if WS, otherwise WS and WSP won't work
    if parameter == "windspeed": # merge columns for WSP and WS
       obs["WSP_PT10M_AVG"] = obs["WSP_PT10M_AVG"].fillna(obs["WS_PT10M_AVG"])

    obs = obs.rename(columns={"fmisid": "station_id"})

    count = len(trad_obs)
    print("Got {} traditional obs stations for time {}".format(count, obstime))

    if count == 0:
        print("Number of observations from Smartmet serve is 0, original MNWC fields are saved")
        write_grib(args, analysistime, fcstime, mnwc)
        sys.exit(1)

    #print(obs.head(5))
    print("min obs:",min(obs.iloc[:,5]))
    print("max obs:",max(obs.iloc[:,5]))
    return obs


def read_netatmo_obs(args, fcstime):
    # read observations for "analysis time" == leadtime 1
    obstime = fcstime[1]

    url = "http://smartmet.fmi.fi/timeseries?producer=NetAtmo&tz=gmt&precision=auto&starttime={}&endtime={}&param=station_id,longitude,latitude,utctime,temperature&format=json&data_quality=1&keyword=snwc".format(
        (obstime - datetime.timedelta(minutes=10)).strftime("%Y%m%d%H%M%S"),
        obstime.strftime("%Y%m%d%H%M%S"),
    )

    resp = requests.get(url)

    crowd_obs = None

    if resp.status_code != 200:
        print("Error fetching NetAtmo data")
    else:
        crowd_obs = resp.json()

    print("Got {} crowd sourced obs stations".format(len(crowd_obs)))

    obs = None

    if crowd_obs is not None:
        obs = pd.DataFrame(crowd_obs)

        # netatmo obs do not contain elevation information, but we need thatn
        # to have the best possible result from optimal interpolation
        #
        # use digital elevation map data to interpolate elevation information
        # to all netatmo station points

        #print("Interpolating elevation to NetAtmo stations")
        dem = xr.open_rasterio(args.dem_data)

        # dem is projected to lambert, our obs data is in latlon
        # transform latlons to projected coordinates

        ll_to_proj = pyproj.Transformer.from_crs("epsg:4326", dem.attrs["crs"])
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
        #print(obs.head(5))
        # reorder/rename columns of the NetAtmo df to match with synop data
        obs = obs[['station_id', 'longitude', 'latitude', 'utctime', 'elevation', 'temperature']]
        obs.rename(columns = {'temperature':'TA_PT1M_AVG'}, inplace = True)
        #print(obs.head(10))
        print("min NetAtmo obs:",min(obs.iloc[:,5]))
        print("max NetAtmo obs:",max(obs.iloc[:,5]))

    return obs


def read_obs(args, fcstime, grid, lc, mnwc):
    """Read observations from smartmet server"""

    # read observations for "analysis" time == leadtime 1
    # obstime = fcstime[1]

    obs = read_conventional_obs(args, fcstime, mnwc)

    # for temperature we also have netatmo stations
    # these are optional


    if args.parameter == "temperature":
        netatmo = read_netatmo_obs(args, fcstime)
        if netatmo is not None:
            obs = pd.concat((obs, netatmo))

        #obs["temperature"] += 273.15

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


def write_grib_message(fp, args, analysistime, forecasttime, data):
    """
    Because there's ~1h delay between the mnwc analysistime and when the data is available for the users,
    the time-parameters for the output data is modified such that new analysistime is +1h and the mnwc data leadtimes are reduced by -1h
    """

    pdtn=0
    tosp=None
    if args.parameter == "humidity":
        levelvalue = 2
        pcat = 1
        pnum = 192
    elif args.parameter == "temperature":
        levelvalue = 2
        pnum = 0
        pcat = 0
    elif args.parameter == "windspeed":
        pcat = 2
        pnum = 1
        levelvalue = 10
    elif args.parameter == "gust":
        levelvalue = 10
        pnum = 22
        pcat = 2
        pdtn=8
        tosp=2
    # Store different time steps as grib msgs
    for j in range(0,len(data)):
        tdata = data[j]
        h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        ecc.codes_set(h, "gridType", "lambert")
        ecc.codes_set(h, "shapeOfTheEarth", 5)
        ecc.codes_set(h, "Nx", tdata.shape[1])
        ecc.codes_set(h, "Ny", tdata.shape[0])
        ecc.codes_set(h, "DxInMetres", 2370000 / (tdata.shape[1] - 1))
        ecc.codes_set(h, "DyInMetres", 2670000 / (tdata.shape[0] - 1))
        ecc.codes_set(h, "jScansPositively", 1)
        ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 50.319616)
        ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0.27828)
        ecc.codes_set(h, "Latin1InDegrees", 63.3)
        ecc.codes_set(h, "Latin2InDegrees", 63.3)
        ecc.codes_set(h, "LoVInDegrees", 15)
        ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
        ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
        ecc.codes_set(h, "dataDate", int(analysistime.strftime("%Y%m%d")))
        ecc.codes_set(h, "dataTime", int(analysistime.strftime("%H%M")))
        ecc.codes_set(
        h, "forecastTime", int((forecasttime[j] - analysistime).total_seconds() / 3600)
        )
        ecc.codes_set(h, "centre", 86)
        ecc.codes_set(h, "generatingProcessIdentifier", 203)
        ecc.codes_set(h, "discipline", 0)
        ecc.codes_set(h, "parameterCategory", pcat)
        ecc.codes_set(h, "parameterNumber", pnum)
        ecc.codes_set(h, "productDefinitionTemplateNumber", pdtn)
        if tosp is not None:
            ecc.codes_set(h, "typeOfStatisticalProcessing", tosp)
        ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
        ecc.codes_set(h, "scaledValueOfFirstFixedSurface", levelvalue)
        ecc.codes_set(h, "packingType", "grid_ccsds")
        ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 0)
        ecc.codes_set(h, "forecastTime", j)
        ecc.codes_set(h, "typeOfGeneratingProcess", 2)  # deterministic forecast
        ecc.codes_set(h, "typeOfProcessedData", 2)  # analysis and forecast products
        ecc.codes_set_values(h, tdata.flatten())
        ecc.codes_write(h, fp)
    ecc.codes_release(h)


def write_grib(args, analysistime, forecasttime, data):

    if args.output.startswith("s3://"):
        openfile = fsspec.open(
            "simplecache::{}".format(args.output),
            "wb",
            s3={
                "anon": False,
                "key": os.environ["S3_ACCESS_KEY_ID"],
                "secret": os.environ["S3_SECRET_ACCESS_KEY"],
                "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"},
            },
        )
        with openfile as fpout:
            write_grib_message(fpout, args, analysistime, forecasttime, data)
    else:
        with open(args.output, "wb") as fpout:
            write_grib_message(fpout, args, analysistime, forecasttime, data)

    print(f"Wrote file {args.output}")


def interpolate_single_time(grid, background, points, obs,obs_to_background_variance_ratio, pobs, structure, max_points, idx, q):

    # perform optimal interpolation
    tmp_output = gridpp.optimal_interpolation(
            grid,
            background,
            points,
            obs[idx]["biasc"].to_numpy(),
            obs_to_background_variance_ratio,
            pobs,
            structure,
            max_points,
        )

    print("step {} min grid: {:.1f} max grid: {:.1f}".format(idx, np.amin(tmp_output), np.amax(tmp_output)))

    if q is not None:
        # return index and output, so that the results can
        # later be sorted correctly
        q.put((idx, tmp_output))
    else:
        return tmp_output


def interpolate(grid, points, background, obs, args, lc):
    """Perform optimal interpolation"""

    output = []

    # create a mask to restrict the modifications only to land area (where lc = 1)
    lc0 = np.logical_not(lc).astype(int)

    # Interpolate background data to observation points
    # When bias is gridded then background is zero so pobs is just array of zeros
    pobs = gridpp.nearest(grid, points, background)

    # Barnes structure function with horizontal decorrelation length 30km,
    # vertical decorrelation length 200m
    structure = gridpp.BarnesStructure(30000, 200, 0.5)

    # Include at most this many "observation points" when interpolating to a grid point
    max_points = 20

    # error variance ratio between observations and background
    # smaller values -> more trust to observations
    obs_to_background_variance_ratio = np.full(points.size(), 0.1)

    if args.disable_multiprocessing:
        output = [interpolate_single_time(grid, background, points, obs, obs_to_background_variance_ratio, pobs, structure, max_points, x, None) for x in range(len(obs))]

    else:
        q = Queue()
        processes = []
        outputd = {}

        for i in range(len(obs)):
            processes.append(Process(target=interpolate_single_time, args=(grid, background, points, obs, obs_to_background_variance_ratio, pobs, structure, max_points, i, q)))
            processes[-1].start()

        for p in processes:
            # get return values from queue
            # they might be in any order (non-consecutive)
            ret = q.get()
            outputd[ret[0]] = ret[1]

        for p in processes:
            p.join()

        for i in range(len(obs)):
            # sort return values from 0 to 8
            output.append(outputd[i])

    return output

def train_data(args, points, obs, grid, topo, ws, rh, t2, wg, cl, ps, wd, q2, forecasttime):
    data = pd.DataFrame()
    tmp_data = pd.DataFrame()

    # Calculate the bias at the leadtime 1h which is used in ML model and for the obs analysis
    if args.parameter == "temperature":
        forecast = gridpp.nearest(grid, points, t2[1])
    elif args.parameter == "humidity":
        forecast = gridpp.nearest(grid, points, rh[1])
    elif args.parameter == "windspeed":
        forecast = gridpp.nearest(grid, points, ws[1])
    elif args.parameter == "gust":
        forecast = gridpp.nearest(grid, points, wg[1])
        #print("havainto:", obs.iloc[:, 5])
    bias = forecast - obs.iloc[:, 5]

    # Calculate the elevation difference between the model and the obs points
    model_elev = gridpp.nearest(grid, points, topo)
    ElevD = model_elev - obs["elevation"]

    # MNWC analystime 0h is not used at all, start from forecasttime 1 (used as observation analysis)
    for j in range(1,len(forecasttime)):
        tmp_data["bias"] = bias
        tmp_data["obs_lat"] = obs["latitude"]
        tmp_data["obs_lon"] = obs["longitude"]
        tmp_data["ElevD"] = ElevD
        tmp_data["S10M"] = gridpp.nearest(grid, points, ws[j])
        tmp_data["D10M"] = gridpp.nearest(grid, points, wd[j])
        tmp_data["GMAX"] = gridpp.nearest(grid, points, wg[j])
        tmp_data["PMSL"] = gridpp.nearest(grid, points, ps[j])
        tmp_data["CCLOW"] = gridpp.nearest(grid, points, cl[j])
        tmp_data["RH2M"] = gridpp.nearest(grid, points, rh[j])
        tmp_data["T2M"] = gridpp.nearest(grid, points, t2[j])
        tmp_data["Q2M"] = gridpp.nearest(grid, points, q2[j])
        tmp_data["leadtime"] = j
        tmp_data["forecasttime"] = forecasttime[j]
        data = data.append(tmp_data,ignore_index=True)
        #data = pd.concat([data,tmp_data],join="inner")
        #print("tmp:",tmp_points)
    return data

# Modify time to sin/cos representation
def encode(data, col, max_val):
    data[col + "_sin"] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col]/max_val)
    return data

# Modify dataframe for xgboost model according parameter
def modify(data, param):
    data = data.assign(month=data.forecasttime.dt.month)
    data = data.assign(hour=data.forecasttime.dt.hour)
    data = encode(data, 'month', 12)
    data = encode(data, 'hour', 24)
    data = data.drop(['month','hour'], axis=1)
    # data = data.dropna()
    if param == "temperature":
        data = data.drop(["GMAX","D10M"], axis=1)
        data = data[['leadtime', 'T2M', 'S10M', 'RH2M', 'PMSL', 'Q2M', 'CCLOW', 'obs_lat',
       'obs_lon', 'ElevD', 'bias', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']]
        data.rename(columns = {'bias':'T1bias'}, inplace = True)
    elif param == "windspeed":
        data = data.drop(["RH2M","Q2M"], axis=1)
        data = data[['leadtime', 'D10M', 'T2M', 'S10M', 'PMSL', 'CCLOW', 'GMAX', 'obs_lat',
       'obs_lon', 'ElevD', 'bias', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']]
        data.rename(columns = {'bias':'WS1bias'}, inplace = True)
    elif param == "gust":
        data = data.drop(["Q2M","CCLOW"], axis=1)
        data = data[['leadtime', 'D10M', 'T2M', 'S10M', 'RH2M', 'PMSL', 'GMAX', 'obs_lat',
       'obs_lon', 'ElevD', 'bias', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']]
        data.rename(columns = {'bias':'WG1bias'}, inplace = True)
    elif param == "humidity":
        data = data.drop(["GMAX","D10M"], axis=1)
        data = data[['leadtime', 'T2M', 'S10M', 'RH2M', 'PMSL', 'Q2M', 'CCLOW', 'obs_lat',
       'obs_lon', 'ElevD', 'bias', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']]
        data.rename(columns = {'bias':'RH1bias'}, inplace = True)

    # modify data precision from f64 to f32
    data[data.select_dtypes(np.float64).columns] = data.select_dtypes(np.float64).astype(np.float32)

    return data

# Calculate ml model point forecast
def ml_forecast(ml_data, param):
    # Load model from S3, I'm unable to make it happen tough...
    if param == "temperature":
        mlname = "T2m"
    elif param == "windspeed":
        mlname = "WS"
    elif param == "gust":
        mlname = "WG"
    elif param == "humidity":
        mlname = "RH"

    regressor = joblib.load('xgb_' + mlname + '_tuned23.joblib') 

    # Check that you have all the leadtimes (0-9)
    ajat = sorted(ml_data['leadtime'].unique().tolist())

    # Leadtime 1 == obs analysis --> ML model not run for this
    lt1 = ml_data[ml_data['leadtime']==1]
    ml_data = ml_data[ml_data.leadtime != 1]

    # Calculate xgboost based bias correction for the leadtimes 2...x
    ml_data["biasc"] = regressor.predict(ml_data)

    if param == "temperature":
        lt1["biasc"] = lt1["T1bias"]
    elif param == "windspeed":
        lt1["biasc"] = lt1["WS1bias"]
    elif param == "gust":
        lt1["biasc"] = lt1["WG1bias"]
    elif param == "humidity":
        lt1["biasc"] = lt1["RH1bias"]

    # Store data to list where each leadtime is it's own list, first leadtime is the obs analysis
    ml_results =  []
    ml_results.append(lt1)

    # Store bias correected forecast for leadtimes 2...x to list
    for j in range(1,len(ajat)):
        lt_tmp = ml_data[ml_data['leadtime']==j+1]
        ml_results.append(lt_tmp)

    return ml_results

def main():
    args = parse_command_line()

    #print("Reading NWP data for", args.parameter )
    st = time.time()
    # read in the parameter which is forecasted
    # background contains mnwc values for different leadtimes
    grid, lons, lats, background, analysistime, forecasttime, lc, topo = read_grid(args)
    # create "zero" background for interpolating the bias
    background0 = copy.copy(background)
    background0[background0 != 0] = 0

    ws, rh, t2, wg, cl, ps, wd, q2 = read_ml_grid(args)
    et = time.time()
    timedif = et-st
    print('Reading NWP data for', args.parameter, 'takes:', round(timedif,1), 'seconds')

    # Read observations from smartmet server
    # Use correct time! == latest obs hour ==  forecasttime[1]

    points, obs = read_obs(args, forecasttime, grid, lc, background)

    ot = time.time()
    timedif = ot-et
    print('Reading OBS data takes:', round(timedif,1) , 'seconds')
    # prepare dataframe for ML code, pandas df
    data = train_data(args, points, obs, grid, topo, ws, rh, t2, wg, cl, ps, wd, q2, forecasttime)

    # preprocess data
    ml_data = modify(data, args.parameter)

    # Produce ML forecast for all the leadtimes
    ml_fcst = ml_forecast(ml_data, args.parameter)
    mlt = time.time()
    timedif = mlt-ot
    print('Producing ML forecasts takes:', round(timedif,1) , 'seconds')
    # Interpolate ML point forecasts for bias correction + 0h analysis time
    diff = interpolate(grid, points, background0[0], ml_fcst, args, lc)
    oit = time.time()
    timedif = oit-mlt
    print('Interpolating forecasts takes:', round(timedif,1) , 'seconds')
    # calculate the final bias corrected forecast fields: MNWC - bias_correction
    # and convert parameter to T-K or RH-0TO1
    output = []
    for j in range(0,len(diff)):
        tmp_output = background[j+1] - diff[j]
        # Implement simple QC thresholds
        if args.parameter == "humidity":
            tmp_output = np.clip(tmp_output, 0, 100)
            tmp_output = tmp_output/100
        elif args.parameter == "windspeed" or args.parameter == "gust":
            tmp_output = np.clip(tmp_output, 0, 50)
        elif args.parameter == "temperature":
            tmp_output = tmp_output + 273.15
        output.append(tmp_output)


    write_grib(args, analysistime, forecasttime, output)

    """
    import matplotlib.pylab as mpl

    # plot diff
    for j in range(0,len(diff)):
        vmin = -5
        vmax = 5
        if args.parameter == "humidity":
             vmin, vmax = -50, 50
        mpl.pcolormesh(lons, lats, diff[j], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        mpl.xlim(0, 35)
        mpl.ylim(55, 75)
        mpl.gca().set_aspect(2)
        mpl.savefig('diff' + args.parameter + str(j) + '.png')
        #mpl.show()
    for k in range(0,len(output)):
        vmin = np.min(output[k])
        vmax = np.max(output[k])
        mpl.pcolormesh(lons, lats, output[k], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        mpl.xlim(0, 35)
        mpl.ylim(55, 75)
        mpl.gca().set_aspect(2)
        mpl.savefig('output' + args.parameter + str(k) + '.png')
        #mpl.show()
    """
    if args.plot:
        plot(obs, background, output, diff, lons, lats, args)


def plot(obs, background, output, diff, lons, lats, args):
    import matplotlib.pyplot as plt

    vmin1 = -5
    vmax1 = 5
    if args.parameter == "temperature":
        obs_parameter = "TA_PT1M_AVG"
        output = list(map(lambda x: x - 273.15, output))
    elif args.parameter == "windspeed":
        obs_parameter = "WSP_PT10M_AVG"
    elif args.parameter == "gust":
        obs_parameter = "WG_PT1H_MAX"
    elif args.parameter == "humidity":
        obs_parameter = "RH_PT1M_AVG"
        output = np.multiply(output,100)
        vmin1 = -30
        vmax1 = 30

    vmin = min(np.amin(background), np.amin(output))
    vmax = min(np.amax(background), np.amax(output))

    #vmin1 =  np.amin(diff)
    #vmax1 =  np.amax(diff)

    for k in range(0,len(diff)):
        plt.figure(figsize=(13, 6), dpi=80)

        plt.subplot(1, 3, 1)
        plt.pcolormesh(
            np.asarray(lons),
            np.asarray(lats),
            background[k+1],
            cmap= "Spectral_r", # "RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(label= "MNWC " + str(k) + "h " + args.parameter , orientation="horizontal")

        plt.subplot(1, 3, 2)
        plt.pcolormesh(
            np.asarray(lons),
            np.asarray(lats),
            diff[k],
            cmap="RdBu_r",
            vmin=vmin1,
            vmax=vmax1,
        )

        """
        plt.scatter(
        obs["longitude"],
        obs["latitude"],
        s=10,
        c=obs[obs_parameter],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        )
        """
        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(label= "Diff " + str(k) + "h " + args.parameter , orientation="horizontal")

        plt.subplot(1, 3, 3)
        plt.pcolormesh(
        np.asarray(lons), np.asarray(lats), output[k], cmap="Spectral_r", vmin=vmin, vmax=vmax
        )

        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label= "XGB " + str(k) + "h " + args.parameter , orientation="horizontal"
        )

        #plt.show()
        plt.savefig('all_' + args.parameter + str(k) + '.png')


if __name__ == "__main__":
    main()
