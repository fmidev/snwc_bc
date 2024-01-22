import numpy as np
import eccodes as ecc
import fsspec
import datetime
import pyproj
import os

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

    if shape == 6:
        return "+R=6371229.0"

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
        uri,
        mode="rb",
        s3={"anon": True, "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"}},
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

    lons = []
    lats = []

    with open(wrk_gribfile, "rb") as fp:
        # print("Reading {}".format(gribfile))

        while True:
            try:
                gh = ecc.codes_grib_new_from_file(fp)
            except ecc.WrongLengthError as e:
                print(e)
                file_stats = os.stat(wrk_gribfile)
                print("Size of {}: {}".format(wrk_gribfile, file_stats.st_size))
                sys.exit(1)

            if gh is None:
                break

            ni = ecc.codes_get_long(gh, "Nx")
            nj = ecc.codes_get_long(gh, "Ny")
            dataDate = ecc.codes_get_long(gh, "dataDate")
            dataTime = ecc.codes_get_long(gh, "dataTime")
            forecastTime = ecc.codes_get_long(gh, "endStep")
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
            return (
                None,
                None,
                np.asarray(values).reshape(nj, ni),
                analysistime,
                forecasttime,
            )
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
    topo = topo / 9.81
    topo = topo[0]
    lc = lc[0]

    if args.parameter == "temperature":
        vals = vals - 273.15
    elif args.parameter == "humidity":
        vals = vals * 100

    grid = gridpp.Grid(lats, lons, topo, lc)
    return grid, lons, lats, vals, analysistime, forecasttime, lc, topo

def write_grib_message(fp, args, analysistime, forecasttime, data):
    pdtn = 70
    tosp = None
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
        pdtn = 72
        tosp = 2
    # Store different time steps as grib msgs
    for j in range(0, len(data)):
        tdata = data[j]
        forecastTime = int((forecasttime[j] - analysistime).total_seconds() / 3600)

        # - For non-aggregated parameters, grib2 key 'forecastTime' is the time of the forecast
        # - For aggregated parameters, it is the start time of the aggregation period. The end of
        #   the period is defined by 'lengthOfTimeRange'
        #   Because snwc is in hourly time steps, reduce forecast time by one

        if tosp == 2:
            forecastTime -= 1

        assert (tosp is None and j + 1 == forecastTime) or (
            tosp == 2 and j == forecastTime
        )
        h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        ecc.codes_set(h, "tablesVersion", 28)
        ecc.codes_set(h, "gridType", "lambert")
        ecc.codes_set(h, "shapeOfTheEarth", 6)
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
        ecc.codes_set(h, "LaDInDegrees", 63.3)
        ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
        ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
        ecc.codes_set(h, "dataDate", int(analysistime.strftime("%Y%m%d")))
        ecc.codes_set(h, "dataTime", int(analysistime.strftime("%H%M")))
        ecc.codes_set(h, "forecastTime", forecastTime)
        ecc.codes_set(h, "centre", 86)
        ecc.codes_set(h, "bitmapPresent", 1)
        ecc.codes_set(h, "generatingProcessIdentifier", 203)
        ecc.codes_set(h, "discipline", 0)
        ecc.codes_set(h, "parameterCategory", pcat)
        ecc.codes_set(h, "parameterNumber", pnum)
        ecc.codes_set(h, "productDefinitionTemplateNumber", pdtn)
        if tosp is not None:
            ecc.codes_set(h, "typeOfStatisticalProcessing", tosp)
            ecc.codes_set(h, "lengthOfTimeRange", 1)
            ecc.codes_set(
                h, "yearOfEndOfOverallTimeInterval", int(forecasttime[j].strftime("%Y"))
            )
            ecc.codes_set(
                h,
                "monthOfEndOfOverallTimeInterval",
                int(forecasttime[j].strftime("%m")),
            )
            ecc.codes_set(
                h, "dayOfEndOfOverallTimeInterval", int(forecasttime[j].strftime("%d"))
            )
            ecc.codes_set(
                h, "hourOfEndOfOverallTimeInterval", int(forecasttime[j].strftime("%H"))
            )
            ecc.codes_set(h, "minuteOfEndOfOverallTimeInterval", 0)
            ecc.codes_set(h, "secondOfEndOfOverallTimeInterval", 0)
        ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
        ecc.codes_set(h, "scaledValueOfFirstFixedSurface", levelvalue)
        ecc.codes_set(h, "packingType", "grid_ccsds")
        ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 1)  # hours
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
