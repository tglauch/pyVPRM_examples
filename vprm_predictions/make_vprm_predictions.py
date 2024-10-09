#!/usr/bin/env python3
import os
import pyVPRM
from pyVPRM.sat_managers.viirs import VIIRS
from pyVPRM.sat_managers.modis import modis
from pyVPRM.sat_managers.copernicus import copernicus_land_cover_map
from pyVPRM.VPRM import vprm
from pyVPRM.meteorologies import era5_monthly_xr, era5_class_dkrz
from pyVPRM.lib.functions import lat_lon_to_modis
from pyVPRM.vprm_models import vprm_modified, vprm_base
import glob
import time
import yaml
import numpy as np
import xarray as xr
import rasterio
import pandas as pd
import pickle
import argparse
import calendar
from datetime import datetime, timedelta
from shapely.geometry import box
import geopandas as gpd
from loguru import logger


def get_hourly_time_range(year, day_of_year):

    start_time = datetime(year, 1, 1) + timedelta(
        days=int(day_of_year) - 1
    )  # Set the starting time based on the day of the year
    end_time = start_time + timedelta(
        hours=1
    )  # Add 1 hour to get the end time of the first hour

    hourly_range = []
    while start_time.timetuple().tm_yday == day_of_year:
        hourly_range.append((start_time))
        start_time = end_time
        end_time = start_time + timedelta(hours=1)
    return hourly_range


# Read command line arguments
p = argparse.ArgumentParser(
    description="Commend Line Arguments", formatter_class=argparse.RawTextHelpFormatter
)
p.add_argument("--h", type=int)
p.add_argument("--v", type=int)
p.add_argument("--config", type=str)
p.add_argument("--n_cpus", type=int, default=1)
p.add_argument("--year", type=int)
args = p.parse_args()
logger.info("Run with args: " + str(args))

h = args.h
v = args.v

# Read config
with open(args.config, "r") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        logger.info(exc)

if not os.path.exists(cfg["predictions_path"]):
    os.makedirs(cfg["predictions_path"])


# Initialize VPRM instance with the copernicus land cover config
vprm_inst = vprm(
    vprm_config_path=os.path.join(
        pyVPRM.__path__[0], "vprm_configs/copernicus_land_cover.yaml"
    ),
    n_cpus=args.n_cpus,
)

# Note: There is no need to convert HDF4 into Netcdf files. You can also use HDF4 files directly.
files = glob.glob(
    os.path.join(
        cfg["sat_image_path"], "*h{:02d}v{:02d}*.nc".format(h, v)  # str(args.year),
    )
)

# Add satellite images to the VPRM instance
for c, i in enumerate(sorted(files)):
    if ".xml" in i:
        continue
    logger.info(i)
    if cfg["satellite"] == "modis":
        handler = modis(sat_image_path=i)
        handler.load()
        vprm_inst.add_sat_img(
            handler,
            b_nir="sur_refl_b02",
            b_red="sur_refl_b01",
            b_blue="sur_refl_b03",
            b_swir="sur_refl_b06",
            which_evi="evi",
            drop_bands=True,
            timestamp_key="sur_refl_day_of_year",
            mask_bad_pixels=True,
            mask_clouds=True,
        )
    else:
        handler = VIIRS(sat_image_path=i)
        handler.load()
        vprm_inst.add_sat_img(
            handler,
            b_nir="SurfReflect_I2",
            b_red="SurfReflect_I1",
            b_blue="no_blue_sensor",
            b_swir="SurfReflect_I3",
            which_evi="evi2",
            drop_bands=True,
        )

# Sort the satellite data by time and run the lowess smoothing
vprm_inst.sort_and_merge_by_timestamp()
vprm_inst.lowess(keys=["evi", "lswi"], times="daily", frac=0.2, it=3)

# Clip EVI and LSWI values to allows ranges
vprm_inst.clip_values("evi", 0, 1)
vprm_inst.clip_values("lswi", -1, 1)

# Calculate the minimum and maximum EVI/LSWI
vprm_inst.calc_min_max_evi_lswi()


# Add land covery map(s) by iterating over all maps in the `copernicus path` and picking those that overlap with our satellite images
lcm = None
for c in glob.glob(os.path.join(cfg["copernicus_path"], "*")):
    # Generate a copernicus_land_cover_map instance
    thandler = copernicus_land_cover_map(c)
    thandler.load()
    bounds = vprm_inst.prototype.sat_img.rio.transform_bounds(thandler.sat_img.rio.crs)

    # Check overlap with our satellite images
    dj = rasterio.coords.disjoint_bounds(bounds, thandler.sat_img.rio.bounds())
    if dj:
        logger.info("Do not add {}".format(c))
        continue
    logger.info("Add {}".format(c))
    if lcm is None:
        lcm = copernicus_land_cover_map(c)
        lcm.load()
    else:
        lcm.add_tile(thandler, reproject=False)

# Crop land cover map to the extend of our satellite images (to speed up computations and save memory)
geom = box(*vprm_inst.sat_imgs.sat_img.rio.bounds())
df = gpd.GeoDataFrame({"id": 1, "geometry": [geom]})
df = df.set_crs(vprm_inst.sat_imgs.sat_img.rio.crs)
df = df.scale(1.3, 1.3)
lcm.crop_to_polygon(df)

# Add land cover map to the VPRM instance. This wil regrid the land cover map to the satellite grid
vprm_inst.add_land_cover_map(
    lcm,
    regridder_save_path=os.path.join(cfg["predictions_path"], "regridder.nc"),
    mpi=False,
)

# Set meteorology
era5_inst = era5_monthly_xr.met_data_handler(
    args.year, 1, 1, 0, "./data/era5", keys=["t2m", "ssrd"]
)

# Load VPRM parameters from a dictionary
with open(cfg["vprm_params_dict"], "rb") as ifile:
    res_dict = pickle.load(ifile)

vprm_model = vprm_base.vprm_base(
    vprm_pre=vprm_inst, met=era5_inst, fit_params_dict=res_dict
)

# Make NEE/GPP flux predictions and save them
days_in_year = 365 + calendar.isleap(args.year)
met_regridder_weights = os.path.join(
    cfg["predictions_path"], "met_regridder_weights.nc"
)

for i in np.arange(160, 161, 1):
    time_range = get_hourly_time_range(int(args.year), i)
    preds_gpp = []
    preds_nee = []
    ts = []
    for t in time_range[:]:
        t0 = time.time()
        logger.info(t)
        pred = vprm_model.make_vprm_predictions(
            t, met_regridder_weights=met_regridder_weights
        )
        if pred is None:
            continue
        preds_gpp.append(pred["gpp"])
        preds_nee.append(pred["nee"])
        ts.append(t)

    preds_gpp = xr.concat(preds_gpp, "time")
    preds_gpp = preds_gpp.assign_coords({"time": ts})
    outpath = os.path.join(
        cfg["predictions_path"],
        "gpp_h{:02d}v{:02d}_{}_{:03d}.h5".format(h, v, args.year, i),
    )
    if os.path.exists(outpath):
        os.remove(outpath)
    preds_gpp.to_netcdf(outpath)
    preds_gpp.close()

    preds_nee = xr.concat(preds_nee, "time")
    preds_nee = preds_nee.assign_coords({"time": ts})
    outpath = os.path.join(
        cfg["predictions_path"],
        "nee_h{:02d}v{:02d}_{}_{:03d}.h5".format(h, v, args.year, i),
    )
    if os.path.exists(outpath):
        os.remove(outpath)
    preds_nee.to_netcdf(outpath)
    preds_nee.close()

logger.info("Done. In order to inspect the output use evaluate_output.ipynb")
