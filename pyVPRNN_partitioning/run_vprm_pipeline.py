#!/usr/bin/env python
"""
End-to-end VPRM training-data pipeline for a single FLUXNET tower site based 
on the FLUXNET shuttle data: https://www.keenangroup.info/fluxnet-data-explorer/.

Stages:
  1. Load FLUXNET half-hourly tower data for the requested site/time range.
  2. Fetch ERA5 boundary-layer height and attach it to the tower data
     (needed by the flux footprint (FFP) model).
  3. Compute the footprint and derive a suitable spatial domain size.
  4. Fetch satellite imagery covering that domain (Sentinel-2 today;
     MODIS pluggable later - see satellite_sources.py).
  5. Compute VPRM satellite indices and gap-fill them with a Kalman filter.
  6. Apply the source's quality masking (e.g. Sentinel-2 SCL clouds/water).
  7. Fetch ESA WorldCover + Copernicus land cover and reconcile classes.
  8. Fetch ERA5-Land meteorology.
  9. Build the final pyvprnn_v1 training dataset and write it to NetCDF.

Usage:
    python run_vprm_pipeline.py --config config.yaml
    python run_vprm_pipeline.py --config config.yaml --site GF-Guy --make-plots
"""

import os
import sys
import glob
import time
import logging
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import yaml
import matplotlib.pyplot as plt
from shapely.geometry import box

import tensorflow as tf
import rioxarray  # noqa: F401  (registers the .rio accessor used on DataArrays/Datasets below)
import stackstac
import planetary_computer as pc
from pystac_client import Client

from utils import (
    km_to_deg,
    plot_footprint_contours,
    site_base_path,
    maybe_extend_syspath,
    retry_with_backoff,
)

import satellite_sources as sat

logger = logging.getLogger("vprm_pipeline")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    p.add_argument("--site", default=None, help="Override site.id from config (e.g. GF-Guy)")
    p.add_argument("--t-start", default=None, help="Override site.t_start (YYYY-MM-DD)")
    p.add_argument("--t-stop", default=None, help="Override site.t_stop (YYYY-MM-DD)")
    p.add_argument("--satellite-source", default=None, choices=["sentinel2", "modis"],
                    help="Override satellite.source from config")
    p.add_argument("--n-cpus", type=int, default=None, help="Override compute.n_cpus")
    p.add_argument("--output-dir", default=None, help="Override paths.output_base_dir")
    p.add_argument("--make-plots", action="store_true", help="Save quicklook + example footprint plots")
    p.add_argument("--overwrite-landcover-cache", action="store_true",
                    help="Force recompute of the land-cover regridding cache even if present")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def load_config(path, args):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if args.site:
        cfg["site"]["id"] = args.site
    if args.t_start:
        cfg["site"]["t_start"] = args.t_start
    if args.t_stop:
        cfg["site"]["t_stop"] = args.t_stop
    if args.satellite_source:
        cfg["satellite"]["source"] = args.satellite_source
    if args.n_cpus:
        cfg["compute"]["n_cpus"] = args.n_cpus
    if args.output_dir:
        cfg["paths"]["output_base_dir"] = args.output_dir
    if args.make_plots:
        cfg["plotting"]["enabled"] = True
    if args.overwrite_landcover_cache:
        cfg["land_cover"]["overwrite_cache"] = True

    return cfg


def get_earthdatahub_token():
    """Single Earthdata Hub token, used for both the footprint BLH pull and full meteorology."""
    token = os.environ.get("EARTHDATAHUB_PAT")
    if not token:
        raise EnvironmentError(
            "Required environment variable 'EARTHDATAHUB_PAT' is not set. See .env.example."
        )
    return token

# ---------------------------------------------------------------------------
# Stage 1-3: flux tower data + footprint
# ---------------------------------------------------------------------------

def load_flux_tower(cfg):
    from pyVPRM.flux_tower_libs.flux_tower_class import fluxnet_shuttle

    site = cfg["site"]["id"]
    pattern = os.path.join(cfg["paths"]["fluxnet_data_dir"], "*", f"*_{site}_FLUXNET_FLUXMET_HH*.csv")
    data_files = glob.glob(pattern)
    if not data_files:
        raise FileNotFoundError(f"No FLUXNET file found for site '{site}' matching: {pattern}")
    if len(data_files) > 1:
        logger.warning("Multiple FLUXNET files matched for %s, using the first: %s", site, data_files[0])

    if "t_start" not in cfg["site"].keys():
        t_start = None
    else:
        t_start = pd.to_datetime(cfg["site"]["t_start"])

    if "t_stop" not in cfg["site"].keys():
        t_stop = None
    else:
        t_stop = pd.to_datetime(cfg["site"]["t_stop"])    
        
    return fluxnet_shuttle(
        data_files[0],
        "SW_IN_F",
        "TA_F",
        need_footprint_variables=True,
        t_start=t_start,
        t_stop=t_stop,
    )

def load_flux_tower_data(flux_tower_inst, cfg, token):
    from pyVPRM.meteorologies.era5_land_destinE_new import met_data_handler

    met_cfg = cfg["advanced"]["meteorology"]
    lat_slice, lon_slice = km_to_deg(flux_tower_inst.lat, flux_tower_inst.lon, km_lat=met_cfg["lat_lon_slice_km"])
    blh_handler = met_data_handler(
        PAT=token,
        keys=met_cfg["footprint_met_vars"],
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        data_product=met_cfg["era5_single_level_product"],
    )

    if cfg['site']['measurement_height'] == 'None':
        measurement_height = None
    else:
        measurement_height = cfg['site']['measurement_height']
        
    flux_tower_inst.add_tower_data(met_inst=blh_handler,
                                   canopy_height_path=cfg['paths']['canopy_height'],
                                   measurement_height=measurement_height)


def compute_footprint(flux_tower_inst, cfg):
    from pyVPRM.flux_tower_libs.FFP_footprint_class import FFP_footprint_manager

    ffp_handler = FFP_footprint_manager(
        time_stamps=flux_tower_inst.flux_data["datetime_utc"],
        flux_tower_manager=flux_tower_inst,
        calculation_grid_side_length=None,
        calculation_grid_pixels_per_side=None,
    )

    distances = ffp_handler.get_fetch_distance(cfg["footprint"]["fetch_percentile"])
    valid = np.isfinite(distances)
    if not valid.any():
        raise RuntimeError("Footprint fetch distances are all non-finite - check tower wind/stability inputs.")

    footprint_size = np.percentile(distances[valid], cfg["footprint"]["fetch_percentile"] * 100)
    logger.info("Footprint size before clipping: %.1f m", footprint_size)
    footprint_size = float(np.clip(footprint_size, cfg["footprint"]["min_size_m"], cfg["footprint"]["max_size_m"]))

    ffp_handler.set_calculation_grid_side_length_and_resolution(
        2 * footprint_size, int((2 * footprint_size) / cfg["advanced"]["footprint"]["ffp_resolution_m"])
    )
    return ffp_handler, footprint_size


# ---------------------------------------------------------------------------
# Stage 4 (optional quicklook)
# ---------------------------------------------------------------------------

def save_rgb_quicklook(cube, cfg, out_path):
    """Sentinel-2-specific preview; skipped for other sources (see satellite_sources.py)."""
    if cfg["satellite"]["source"] != "sentinel2":
        logger.info("RGB quicklook only implemented for sentinel2 - skipping.")
        return

    from pyVPRM.lib.fancy_plot import newfig

    ind = sat.rgb_quicklook_index(cube, cfg)
    rgb = np.dstack([cube.isel(time=ind)[b].values for b in ("red", "green", "blue")]).astype(float)
    rgb = np.clip(rgb, 0, None)
    p_low, p_high = np.nanpercentile(rgb, (5, 96))
    rgb = np.clip((rgb - p_low) / (p_high - p_low), 0, 1) ** (1 / 1.25)

    fig, ax = newfig(1.0, 1.0)
    ax.imshow(rgb, zorder=10)
    ax.axis("off")
    fig.savefig(out_path, dpi=cfg["advanced"]["plotting"]["quicklook_dpi"], bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 5-6: VPRM satellite index processing + quality masking
# ---------------------------------------------------------------------------

def run_vprm_satellite_processing(cube, flux_tower_inst, footprint_size, cfg):
    import pyVPRM
    from pyVPRM.VPRM import vprm_preprocessor

    s_cfg = sat.source_cfg(cfg)
    handler = sat.get_satellite_handler(cube, cfg)
    handler.crop(
        lonlat=(flux_tower_inst.lon, flux_tower_inst.lat),
        radius=footprint_size / 1000 * s_cfg["crop_buffer_factor"],
    )

    vprm_inst = vprm_preprocessor(
        vprm_config_path=os.path.join(pyVPRM.__path__[0], "vprm_configs", "pyvprnn.yaml"),
        n_cpus=cfg["compute"]["n_cpus"],
    )

    band_map = s_cfg["band_map"]
    vprm_inst.add_sat_img(
        handler,
        b_nir=band_map["nir"],
        b_red=band_map["red"],
        b_blue=band_map["blue"],
        b_swir=band_map["swir"],
        b_red_edge=band_map["red_edge"],
        satellite_indices=s_cfg["indices"],
        mask_bad_pixels=True,
        mask_clouds=True,
        mask_snow=True,
        mask_water=True,
        drop_bands=[b for b in s_cfg["bands"] if b != s_cfg.get("quality_band", "scl")],
    )

    vprm_inst.sort_and_merge_by_timestamp(min_length_snow_period=None)

    kalman_cfg = cfg["advanced"]["kalman"]
    t0 = time.time()
    vprm_inst.kalman(
        s_cfg["indices"],
        times="daily",
        transition_covariance=kalman_cfg["transition_covariance"],
        observation_covariance=kalman_cfg["observation_covariance"],
        n_cpus=cfg["compute"]["n_cpus"],
    )
    logger.info("Kalman gap-filling took %.1f s", time.time() - t0)

    return handler, vprm_inst


def save_nirv_quicklook(vprm_inst, cfg, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    vprm_inst.sat_imgs.sat_img.isel({"time_gap_filled": 40})["nirv"].plot.imshow(cmap="Greens", vmin=0.05, vmax=0.4)
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(out_path, dpi=cfg["advanced"]["plotting"]["quicklook_dpi"], bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 7: land cover
# ---------------------------------------------------------------------------

def fetch_esa_worldcover(bbox, cfg):
    from pyVPRM.sat_managers.esa_world_cover import esa_world_cover
 
    lc_cfg = cfg["advanced"]["land_cover"]
    catalog = Client.open(lc_cfg["esa_worldcover_stac_endpoint"])
 
    item = retry_with_backoff(
        lambda: next(catalog.search(collections=["esa-worldcover"], bbox=bbox, max_items=1).items()),
        description="ESA WorldCover STAC search",
    )
    item = pc.sign(item)
 
    wc = stackstac.stack(item, assets=["map"], bounds_latlon=bbox)
    retry_with_backoff(wc.load, description="ESA WorldCover tile load")
 
    wc_fixed = (
        wc.drop_vars(
            ["description", "instruments", "proj:shape", "proj:transform", "raster:bands", "proj:epsg", "title", "epsg"]
        )
        .squeeze(drop=True)
        .to_dataset(name="band_1")
    )
    wc_fixed.rio.write_crs(wc.rio.crs, inplace=True)
    wc_fixed.rio.write_transform(wc.rio.transform(), inplace=True)
 
    return esa_world_cover(sat_image_path=None, sat_img=wc_fixed)



def fetch_copernicus_land_cover(vprm_inst, cfg):
    from pyVPRM.sat_managers.copernicus import copernicus_land_cover_map
    import pyVPRM

    lc_cfg = cfg["advanced"]["land_cover"]
    lcm2 = copernicus_land_cover_map(cfg["paths"]["copernicus_land_cover_tif"])
    lcm2.load()

    geom = box(*vprm_inst.sat_imgs.sat_img.rio.bounds())
    df = gpd.GeoDataFrame({"id": 1, "geometry": [geom]}).set_crs(vprm_inst.sat_imgs.sat_img.rio.crs)
    scale = lc_cfg["copernicus_geom_scale_factor"]
    df = df.scale(scale, scale).to_crs("WGS84")
    b = df.geometry.bounds.iloc[0]

    lcm2.sat_img = lcm2.sat_img.rio.clip_box(minx=b["minx"], miny=b["miny"], maxx=b["maxx"], maxy=b["maxy"])
    lcm2.map_veg_classes(os.path.join(pyVPRM.__path__[0], "vprm_configs", "copernicus_land_cover.yaml"))
    lcm2.map_class_to_nearest_valid_class()
    return lcm2


def reconcile_land_cover_classes(lcm, lcm2, flux_tower_inst, cfg):
    """
    Hybrid-map step: ESA WorldCover's generic "tree cover" class is split
    into evergreen/deciduous using the Copernicus map, for forested sites
    only. See advanced.land_cover.hybrid_reconciliation in config.yaml -
    delete both if you move to a single land-cover source later.
    """
    hybrid_cfg = cfg["advanced"]["land_cover"]["hybrid_reconciliation"]
    tree_cover = hybrid_cfg["esa_tree_cover_class"]
    evergreen = hybrid_cfg["vprm_evergreen_class"]
    deciduous = hybrid_cfg["vprm_deciduous_class"]

    lcm2_high_res = lcm2.sat_img.sel(
        x=lcm.sat_img.coords["x"].values, y=lcm.sat_img.coords["y"].values, method="nearest"
    ).assign_coords(x=lcm.sat_img.x, y=lcm.sat_img.y)

    land_cover_type = flux_tower_inst.land_cover_type
    if land_cover_type in ("ENF", "EBF"):
        lcm.sat_img["band_1"] = xr.where(lcm.sat_img["band_1"] == tree_cover, evergreen, lcm.sat_img["band_1"])
        lcm.sat_img["band_1"] = xr.where(
            (lcm.sat_img["band_1"] == evergreen) & (lcm2_high_res["band_1"] == deciduous),
            deciduous,
            lcm.sat_img["band_1"],
        )
    elif land_cover_type == "DBF":
        lcm.sat_img["band_1"] = xr.where(lcm.sat_img["band_1"] == tree_cover, deciduous, lcm.sat_img["band_1"])
        lcm.sat_img["band_1"] = xr.where(
            (lcm.sat_img["band_1"] == deciduous) & (lcm2_high_res["band_1"] == evergreen),
            evergreen,
            lcm.sat_img["band_1"],
        )
    else:
        logger.info("Land cover type '%s' - no evergreen/deciduous reconciliation applied. Change station IGBP to either EF or DF", land_cover_type)
    return lcm2_high_res


def prepare_land_cover_output_paths(base_path, cfg):
    veg_file = os.path.join(base_path, "lcm_out.nc")
    regridder_path = os.path.join(base_path, "lcm_regridder.nc")
    os.makedirs(base_path, exist_ok=True)

    if cfg["land_cover"]["overwrite_cache"]:
        for f in (veg_file, regridder_path):
            if os.path.exists(f):
                os.remove(f)
    elif os.path.exists(veg_file) and os.path.exists(regridder_path):
        logger.info("Land-cover cache already exists at %s - reusing (use --overwrite-landcover-cache to force).", base_path)

    return veg_file, regridder_path


# ---------------------------------------------------------------------------
# Stage 8: meteorology
# ---------------------------------------------------------------------------

def build_meteo_transforms(cfg):
    """Translate the config's declarative meteo-variable ops into callables."""
    ops = {"none": lambda x: x, "sub_kelvin": lambda x: x - 273.15}

    def make_fn(spec):
        if spec["op"] == "div":
            return lambda x, v=float(spec["value"]): x / v
        if spec["op"] == "mul":
            return lambda x, v=float(spec["value"]): x * v
        return ops[spec["op"]]

    return {name: make_fn(spec) for name, spec in cfg["advanced"]["meteorology"]["full_met_vars"].items()}


def fetch_full_meteorology(flux_tower_inst, cfg, token):
    from pyVPRM.meteorologies.era5_land_destinE_new import met_data_handler

    met_cfg = cfg["advanced"]["meteorology"]
    lat_slice, lon_slice = km_to_deg(flux_tower_inst.lat, flux_tower_inst.lon, km_lat=met_cfg["lat_lon_slice_km"])
    return met_data_handler(
        PAT=token,
        keys=list(met_cfg["full_met_vars"].keys()),
        lat_slice=lat_slice,
        lon_slice=lon_slice,
    )


# ---------------------------------------------------------------------------
# Stage 9: final training dataset
# ---------------------------------------------------------------------------

def build_training_dataset(vprm_inst, met_inst, ffp_handler, flux_tower_inst, base_path, meteo_vars, cfg):
    from pyVPRM.vprm_models.pyvprnn_v1 import pyvprnn_v1

    pyvprnn_inst = pyvprnn_v1()
    pyvprnn_inst.get_training_data(
        vprm_pre=vprm_inst,
        met=met_inst,
        footprint=ffp_handler,
        flux_tower=flux_tower_inst,
        base_path=base_path,
        meteo_vars=meteo_vars,
        n_chunks=cfg["compute"]["n_chunks"],
    )
    pyvprnn_inst.ds.attrs["spec"] = ""
    return pyvprnn_inst


def save_training_dataset(pyvprnn_inst, out_path):
    coords_to_drop = [c for c in pyvprnn_inst.ds.coords if c not in pyvprnn_inst.ds.dims and c != "days_since_t0"]
    to_save = pyvprnn_inst.ds.reset_coords(coords_to_drop, drop=True)
    for v in to_save.variables.values():
        v.encoding = {}
    to_save.to_netcdf(out_path)
    logger.info("Wrote training dataset to %s", out_path)


def make_example_footprint_plot(pyvprnn_inst, cfg, out_path):
    """
    Save an example EVI + footprint-contour overlay. Automatically finds the
    first index for which a matching footprint timestamp exists, rather than
    relying on a hardcoded row index (which breaks for other sites/time ranges).
    """
    from pyVPRM.lib.fancy_plot import newfig

    ffp_times = set(pyvprnn_inst.ds["ffp_footprint"]["t"].values)
    ind = None
    for i in range(len(pyvprnn_inst.ds["t2m"]["datetime_utc"]) - 3):
        valid_time = pyvprnn_inst.ds["t2m"].isel(datetime_utc=i)["valid_time"].values
        valid_time_next = pyvprnn_inst.ds["t2m"].isel(datetime_utc=i + 3)["valid_time"].values
        if valid_time in ffp_times and valid_time_next in ffp_times:
            ind = i
            break

    if ind is None:
        logger.warning("No timestamp with a matching footprint found - skipping example plot.")
        return

    fig, ax = newfig(0.9, 0.8)
    days = pyvprnn_inst.ds["t2m"].isel(datetime_utc=ind)["days_since_t0"]
    pyvprnn_inst.ds["evi"].sel(time_gap_filled=days).plot.imshow(cmap="Greens", vmin=0.3, vmax=0.9)

    plot_footprint_contours(
        ax, pyvprnn_inst.ds["ffp_footprint"].sel(t=pyvprnn_inst.ds["t2m"].isel(datetime_utc=ind)["valid_time"]), color="red"
    )
    plot_footprint_contours(
        ax, pyvprnn_inst.ds["ffp_footprint"].sel(t=pyvprnn_inst.ds["t2m"].isel(datetime_utc=ind + 3)["valid_time"]), color="blue"
    )
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(out_path, dpi=cfg["advanced"]["plotting"]["example_plot_dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved example footprint plot to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    
    cfg = load_config(args.config, args)
    maybe_extend_syspath(cfg)
    token = get_earthdatahub_token()

    import socket
    logger.info("Running on %s (pid %d)", socket.gethostname(), os.getpid())
    logger.info("GPUs: %s", tf.config.list_physical_devices("GPU"))

    site = cfg["site"]["id"]

    # --- flux tower + footprint --------------------------------------------------
    flux_tower_inst = load_flux_tower(cfg)
    base_path = site_base_path(cfg["paths"]["output_base_dir"], flux_tower_inst.land_cover_type, site)
    os.makedirs(base_path, exist_ok=True)
    figure_dir = os.path.join(base_path, 'figures')
    os.makedirs(figure_dir, exist_ok=True)

    load_flux_tower_data(flux_tower_inst, cfg, token)

    ffp_handler, footprint_size = compute_footprint(flux_tower_inst, cfg)
    logger.info("Site %s | footprint size %.0f m | satellite source %s",
                site, footprint_size, cfg["satellite"]["source"])

    # --- satellite imagery ---------------------------------------------------
    cube, bbox, point = sat.fetch_satellite_stack(flux_tower_inst, footprint_size, cfg)
    if cfg["plotting"]["enabled"]:
        save_rgb_quicklook(cube, cfg, os.path.join(figure_dir,
                                                   "rgb_quicklook.png"))

    handler, vprm_inst = run_vprm_satellite_processing(cube, flux_tower_inst, footprint_size, cfg)
    sat.mask_satellite(handler, vprm_inst, cfg)
    if cfg["plotting"]["enabled"]:
        save_nirv_quicklook(vprm_inst, cfg, os.path.join(figure_dir,
                                                         "nirv_quicklook.png"))

    # --- land cover ------------------------------------------------------------
    veg_file, regridder_path = prepare_land_cover_output_paths(base_path, cfg)
    lcm = fetch_esa_worldcover(bbox, cfg)
    lcm2 = fetch_copernicus_land_cover(vprm_inst, cfg)
    reconcile_land_cover_classes(lcm, lcm2, flux_tower_inst, cfg)

    vprm_inst.add_land_cover_map(lcm, regridder_save_path=regridder_path, mpi=False)
    vprm_inst.land_cover_type.save(veg_file)

    # --- meteorology + final dataset -------------------------------------------
    met_inst = fetch_full_meteorology(flux_tower_inst, cfg, token)
    meteo_vars = build_meteo_transforms(cfg)

    pyvprnn_inst = build_training_dataset(vprm_inst, met_inst, ffp_handler, flux_tower_inst, base_path, meteo_vars, cfg)
    save_training_dataset(pyvprnn_inst, os.path.join(base_path, cfg["paths"]['training_data_filename']))

    if cfg["plotting"]["enabled"]:
        make_example_footprint_plot(pyvprnn_inst, cfg, os.path.join(figure_dir,
                                                                    f"{site}_example.png"))


if __name__ == "__main__":
    main()
