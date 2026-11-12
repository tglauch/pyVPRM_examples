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

Quality control (QCLog): tower meteorology is checked for unit errors (unambiguous ones are
corrected, e.g. VPD in Pa instead of hPa), land-cover reconciliation is logged per class, and
the final dataset is checked (land-cover fractions sum to 1, footprint coverage, satellite gaps,
tower vs ERA5 temperature). All findings are printed at the end and stored in the NetCDF
attributes "qc_log" (JSON list) and "qc_status" (ok / warning / error).

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

import json

import satellite_sources as sat

logger = logging.getLogger("vprm_pipeline")


# ---------------------------------------------------------------------------
# Quality-control log
# ---------------------------------------------------------------------------

class QCLog:
    """Collects checks and corrections; printed at the end and written to the NetCDF attributes."""
    LEVELS = ("info", "correction", "warning", "error")

    def __init__(self):
        self.entries = []

    def add(self, level, stage, message, **values):
        assert level in self.LEVELS, level
        entry = dict(level=level, stage=stage, message=message,
                     **{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in values.items()})
        self.entries.append(entry)
        log = {"info": logger.info, "correction": logger.warning, "warning": logger.warning, "error": logger.error}[level]
        log("[QC %s] %s: %s %s", level, stage, message, values if values else "")

    @property
    def status(self):
        lv = {e["level"] for e in self.entries}
        return "error" if "error" in lv else "warning" if ("warning" in lv or "correction" in lv) else "ok"

    def summary(self):
        lines = [f"QC status: {self.status} ({len(self.entries)} entries)"]
        for lev in ("error", "warning", "correction", "info"):
            for e in self.entries:
                if e["level"] == lev:
                    extra = {k: v for k, v in e.items() if k not in ("level", "stage", "message")}
                    lines.append(f"  [{lev:10s}] {e['stage']}: {e['message']}" + (f"  {extra}" if extra else ""))
        return "\n".join(lines)

    def to_attrs(self, ds):
        ds.attrs["qc_status"] = self.status
        ds.attrs["qc_log"] = json.dumps(self.entries, default=str)
        return ds


QC = QCLog()


def _qc_findings(findings, stage, qc=QC):
    for f in findings:
        details = {k: v for k, v in f.items() if k not in ("level", "var", "msg")}
        qc.add(f["level"], stage, f"{f['var']}: {f['msg']}", **details)


def check_tower_meteorology(flux_tower_inst, qc=QC):
    """Unit / range / consistency checks on the FLUXNET tower data before anything is derived from
    it (footprint, RH, diffusion bound); see site_checks.py. Units are judged per day and month,
    so a unit change within the record is found. Unambiguous problems are corrected in place
    (original kept as <var>_orig), everything else is logged."""
    from site_checks import check_dataframe
    df, findings = check_dataframe(flux_tower_inst.flux_data, lon=flux_tower_inst.lon)
    flux_tower_inst.flux_data = df
    _qc_findings(findings, "tower", qc)


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
    p.add_argument("--no_footprint", action="store_true",
                    help="Do not calculate the footprint in input data generation")
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
    import pyVPRM
    print(pyVPRM.__path__)
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
 
    # all tiles that intersect the box (a domain on a 3-degree tile edge needs two or four tiles);
    # only the newest map version, tiles do not overlap within one version
    items = retry_with_backoff(
        lambda: list(catalog.search(collections=["esa-worldcover"], bbox=bbox).items()),
        description="ESA WorldCover STAC search",
    )
    if not items:
        raise RuntimeError(f"No ESA WorldCover tile found for bbox {bbox}")
    newest = max(i.properties.get("esa_worldcover:product_version", "") for i in items)
    items = [pc.sign(i) for i in items if i.properties.get("esa_worldcover:product_version", "") == newest]
    QC.add("info", "land cover", f"ESA WorldCover: {len(items)} tile(s), version {newest or 'unknown'}")

    wc = stackstac.stack(items, assets=["map"], bounds_latlon=bbox)
    retry_with_backoff(wc.load, description="ESA WorldCover tile load")
    if "time" in wc.dims and wc.sizes["time"] > 1:
        wc = wc.max("time", skipna=True).expand_dims(time=1)             # mosaic of non-overlapping tiles
    nodata = float(((wc == 0) | wc.isnull()).mean())
    if nodata > 0.01:
        QC.add("warning", "land cover", "ESA WorldCover has no data in part of the domain", nodata_share=nodata)
 
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


# default type of ESA "tree cover" pixels that Copernicus cannot classify as evergreen / deciduous,
# by site IGBP class: evergreen everywhere except deciduous forest sites (one class per site; a
# constant tree type carries no information for training a single site, the trees only have to be
# counted as forest). Copernicus forest types replace the default wherever they exist;
# site.tree_type in the config overrides the default.
TREE_TYPE_DEFAULT = {"DBF": "deciduous", "DNF": "deciduous"}


def reconcile_land_cover_classes(lcm, lcm2, flux_tower_inst, cfg, qc=QC):
    """
    Hybrid-map step: ESA WorldCover's generic "tree cover" class is split into evergreen /
    deciduous using the Copernicus map, for EVERY site type (previously only ENF / EBF / DBF / MF;
    at all other sites the tree pixels kept the ESA code, which is no pyVPRNN class, and were
    lost - e.g. the trees of savannas, woodlands and hedges in grasslands).
    Tree pixels get evergreen by default (deciduous at DBF / DNF sites, or site.tree_type from the
    config); wherever Copernicus gives a forest type (evergreen / deciduous), that type is used.
    """
    hybrid_cfg = cfg["advanced"]["land_cover"]["hybrid_reconciliation"]
    tree_cover = hybrid_cfg["esa_tree_cover_class"]
    evergreen = hybrid_cfg["vprm_evergreen_class"]
    deciduous = hybrid_cfg["vprm_deciduous_class"]

    lcm2_high_res = lcm2.sat_img.sel(
        x=lcm.sat_img.coords["x"].values, y=lcm.sat_img.coords["y"].values, method="nearest"
    ).assign_coords(x=lcm.sat_img.x, y=lcm.sat_img.y)

    band = lcm.sat_img["band_1"]
    cop = lcm2_high_res["band_1"]
    land_cover_type = flux_tower_inst.land_cover_type
    default = cfg["site"].get("tree_type") or TREE_TYPE_DEFAULT.get(land_cover_type, "evergreen")
    if default not in ("evergreen", "deciduous"):
        raise ValueError(f"site.tree_type must be 'evergreen' or 'deciduous', got {default!r}")

    is_tree = band == tree_cover
    cop_ever, cop_deci = is_tree & (cop == evergreen), is_tree & (cop == deciduous)
    untyped = is_tree & ~cop_ever & ~cop_deci
    fill = xr.full_like(band, evergreen if default == "evergreen" else deciduous)
    new = xr.where(cop_ever, evergreen, band)
    new = xr.where(cop_deci, deciduous, new)
    new = xr.where(untyped, fill, new)
    lcm.sat_img["band_1"] = new.astype(band.dtype)

    n_tree = int(is_tree.sum())
    valid = int((band.notnull() & (band > 0)).sum())
    if n_tree:
        qc.add("info", "land cover",
               f"ESA tree pixels ({n_tree / max(valid, 1):.1%} of the domain): Copernicus evergreen "
               f"{int(cop_ever.sum()) / n_tree:.1%}, deciduous {int(cop_deci.sum()) / n_tree:.1%}, "
               f"untyped -> {default} {int(untyped.sum()) / n_tree:.1%} (site type {land_cover_type})")
    vals, counts = np.unique(band.values[np.isfinite(band.values)], return_counts=True)
    qc.add("info", "land cover", "ESA class shares: " + ", ".join(
        f"{int(v)}: {c / counts.sum():.1%}" for v, c in zip(vals, counts) if c / counts.sum() >= 0.005))
    return lcm2_high_res


def land_cover_fraction_sum(lc_obj):
    """Sum over classes of a regridded land-cover object (Dataset with one variable per class, or a
    DataArray with a class dimension). Returns a DataArray or None."""
    img = getattr(lc_obj, "sat_img", lc_obj)
    try:
        if isinstance(img, xr.Dataset):
            vars_ = [v for v in img.data_vars if img[v].ndim >= 2]
            if len(vars_) == 1 and img[vars_[0]].ndim == 3:
                da = img[vars_[0]]
                cdim = [d for d in da.dims if d not in ("x", "y", "lat", "lon")][0]
                return da.sum(cdim, skipna=True)
            return sum(img[v].fillna(0) for v in vars_)
        cdim = [d for d in img.dims if d not in ("x", "y", "lat", "lon")][0]
        return img.sum(cdim, skipna=True)
    except Exception as err:                                            # unknown structure: skip the check
        logger.info("land-cover fraction check skipped: %s", err)
        return None


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
    print('ffp handler 2', ffp_handler)
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


def check_final_dataset(ds, qc=QC, n_fp_samples=200):
    """Checks on the dataset as it is written (what the model will actually see)."""
    # land-cover fractions, domain and footprint-weighted
    if "land_cover_map" in ds:
        lc = ds["land_cover_map"]
        cdim = [d for d in lc.dims if d not in ("x", "y", "lat", "lon")][0]
        tot = lc.sum(cdim, skipna=True)
        med, low = float(tot.median()), float((tot < 0.95).mean())
        fp_sum = np.nan
        if "ffp_footprint" in ds:
            tdim = [d for d in ds["ffp_footprint"].dims if d not in tot.dims][0]
            step = max(ds.sizes[tdim] // n_fp_samples, 1)
            fp = ds["ffp_footprint"].isel({tdim: slice(0, None, step)}).mean(tdim)
            w = (fp / fp.sum()).fillna(0)
            fp_sum = float((tot * w).sum())
        level = "error" if (med < 0.9 or fp_sum < 0.9) else "warning" if (low > 0.05 or fp_sum < 0.98) else "info"
        qc.add(level, "dataset", "land-cover fractions: sum over classes", domain_median=med,
               share_pixels_below_0_95=low, footprint_weighted=fp_sum)
        shares = {str(c): float(lc.sel({cdim: c}).mean()) for c in lc[cdim].values}
        qc.add("info", "dataset", "domain-mean land-cover fractions per class", **shares)
    else:
        qc.add("error", "dataset", "no land_cover_map in the dataset")

    # satellite indices: gaps after the Kalman filter
    for v in ("nirv", "evi", "lswi"):
        if v in ds:
            nan = float(ds[v].isnull().mean())
            if nan > 0.05:
                qc.add("warning", "dataset", f"{v}: missing values after gap filling", nan_share=nan)

    # tower vs ERA5 temperature (unit or site-position errors show up as large offsets)
    def _series(name):
        if name not in ds or "datetime_utc" not in ds[name].dims:
            return None
        da = ds[name]
        extra = [d for d in da.dims if d != "datetime_utc"]
        if extra:
            da = da.mean(extra, skipna=True)
        x = pd.Series(np.asarray(da.values, float))
        return x.where(x > -9990)
    ta, t5 = _series("t2m"), _series("t2m_era5")
    if ta is not None and t5 is not None:
        if np.nanmedian(t5) > 150:
            t5 = t5 - 273.15
        if np.nanmedian(ta) > 150:
            qc.add("error", "dataset", "tower t2m looks like Kelvin", median=float(np.nanmedian(ta)))
        else:
            d = float(np.nanmean(ta - t5))
            r = float(pd.concat([ta, t5], axis=1).corr().iloc[0, 1])
            qc.add("warning" if (abs(d) > 4 or r < 0.8) else "info", "dataset",
                   "tower vs ERA5 air temperature", mean_difference_K=d, correlation=r)

    # derived relative humidity (zero / saturated plateaus point to VPD unit errors)
    rh = _series("RH_from_VPD")
    if rh is not None and rh.notna().any():
        scale = 100.0 if np.nanmedian(rh) > 1.5 else 1.0
        zero, sat_ = float((rh <= 0.001 * scale).mean()), float((rh >= 0.999 * scale).mean())
        qc.add("error" if zero > 0.2 else "warning" if (zero > 0.02 or sat_ > 0.3) else "info", "dataset",
               "RH derived from VPD", median=float(np.nanmedian(rh)), share_zero=zero, share_saturated=sat_)

    # ERA5 soil moisture and radiation plausibility
    for v, (lo, hi) in {"swvl1_era5": (0.0, 0.8), "swvl2_era5": (0.0, 0.8)}.items():
        x = _series(v)
        if x is not None and x.notna().any() and not (lo <= np.nanmedian(x) <= hi):
            qc.add("warning", "dataset", f"{v} median outside {lo}..{hi}", median=float(np.nanmedian(x)))
    return ds


def save_training_dataset(pyvprnn_inst, out_path):
    from derived_variables import prepare_site_dataset
    from site_checks import format_findings
    coords_to_drop = [c for c in pyvprnn_inst.ds.coords if c not in pyvprnn_inst.ds.dims and c != "days_since_t0"]
    to_save = pyvprnn_inst.ds.reset_coords(coords_to_drop, drop=True)
    # units / consistency of everything the model reads (tower, ERA5, satellite, targets),
    # derived variables (RH_from_VPD) from the checked inputs, model inputs present
    to_save, findings, ok = prepare_site_dataset(to_save)
    _qc_findings(findings, "dataset")
    if not ok:
        bar = "!" * 100
        print(f"\n{bar}\n!!! SITE CHECKS FAILED - the file is written, but do not train on it before "
              f"the errors are resolved:\n{format_findings(findings, levels=('error',))}\n{bar}\n")
    check_final_dataset(to_save)
    QC.to_attrs(to_save)
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
    check_tower_meteorology(flux_tower_inst)

    if args.no_footprint:
        ffp_handler, footprint_size = None, 200
    else:
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

    cache_reused = os.path.exists(regridder_path)
    vprm_inst.add_land_cover_map(lcm, regridder_save_path=regridder_path, mpi=False)
    tot = land_cover_fraction_sum(vprm_inst.land_cover_type)
    if tot is not None:
        med = float(tot.median())
        if med < 0.95 and cache_reused:
            QC.add("correction", "land cover", "land-cover fractions do not sum to 1 with the cached regridder "
                   "(stale cache from another domain?) - recomputed", median_sum_before=med)
            for f in (veg_file, regridder_path):
                if os.path.exists(f):
                    os.remove(f)
            vprm_inst.add_land_cover_map(lcm, regridder_save_path=regridder_path, mpi=False)
            tot = land_cover_fraction_sum(vprm_inst.land_cover_type)
            med = float(tot.median()) if tot is not None else np.nan
        low = float((tot < 0.95).mean()) if tot is not None else np.nan
        QC.add("error" if med < 0.9 else "warning" if low > 0.05 else "info", "land cover",
               "regridded land-cover fractions: median sum over classes", median_sum=med, share_below_0_95=low)
    vprm_inst.land_cover_type.save(veg_file)

    # --- meteorology + final dataset -------------------------------------------
    met_inst = fetch_full_meteorology(flux_tower_inst, cfg, token)
    meteo_vars = build_meteo_transforms(cfg)
    
    print('FFP handler', ffp_handler)
    pyvprnn_inst = build_training_dataset(vprm_inst, met_inst, ffp_handler, flux_tower_inst, base_path, meteo_vars, cfg)

    if args.no_footprint:
        opath = os.path.join(base_path, cfg["paths"]['inference_data_filename'])
    else:
        opath = os.path.join(base_path, cfg["paths"]['training_data_filename'])
    save_training_dataset(pyvprnn_inst, opath)

    if cfg["plotting"]["enabled"]:
        make_example_footprint_plot(pyvprnn_inst, cfg, os.path.join(figure_dir,
                                                                    f"{site}_example.png"))

    summary = QC.summary()
    print("\n" + "=" * 80 + f"\n{site}: " + summary + "\n" + "=" * 80)
    with open(os.path.join(base_path, "qc_log.txt"), "w") as f:
        f.write(f"{site}\n{summary}\n")


if __name__ == "__main__":
    main()