"""
Satellite data source implementations.

The rest of the pipeline talks to whichever source is selected in
config.yaml (satellite.source) only through the two functions below -
it never assumes Sentinel-2-specific things like band names or the SCL
layer. That's what makes swapping in MODIS later a matter of filling in
this file rather than touching run_vprm_pipeline.py.

Each source must provide:

  fetch(flux_tower_inst, footprint_size, cfg) -> (cube, bbox, point)
      cube must be an xr.Dataset with one variable per band in
      advanced.satellite.<source>.bands, chunked over "time", already
      cropped/scaled/CRS-tagged - i.e. ready to hand to sentinel2()-like
      wrapper classes in pyVPRM.sat_managers.

  mask(handler, vprm_inst, cfg) -> None
      Applies in-place quality masking to vprm_inst.sat_imgs.sat_img
      (e.g. Sentinel-2 SCL clouds/water; MODIS would use its own QA/state
      bands). Also responsible for calling vprm_inst.calc_min_max_evi_lswi().

To add a new source:
  1. Add a advanced.satellite.<name> block to config.yaml (band_map,
     bands, indices, endpoint, quality-masking settings).
  2. Implement fetch_<name>_stack() and mask_<name>() below.
  3. Register both in SOURCES.
  4. Set satellite.source: "<name>" in config.yaml.
"""

import logging
from datetime import timedelta

import numpy as np
import xarray as xr
import geopandas as gpd
import stackstac
from pystac_client import Client
from shapely.geometry import Point

from utils import make_bbox

logger = logging.getLogger("vprm_pipeline")


def source_cfg(cfg):
    """The advanced-settings block for whichever source is currently selected."""
    return cfg["advanced"]["satellite"][cfg["satellite"]["source"]]


# ---------------------------------------------------------------------------
# Sentinel-2
# ---------------------------------------------------------------------------

def _find_mgrs_tile(point, shapefile_path):
    tiles_gdf = gpd.read_file(shapefile_path)
    matches = [row.Name for _, row in tiles_gdf.iterrows() if row.geometry.contains(point)]
    if not matches:
        raise RuntimeError("No Sentinel-2 MGRS tile found for site location.")
    if len(matches) > 1:
        logger.warning("Point falls in multiple MGRS tiles %s, using the first.", matches)
    return matches[0]


def fetch_sentinel2_stack(flux_tower_inst, footprint_size, cfg):
    s2_cfg = source_cfg(cfg)
    lat, lon = flux_tower_inst.lat, flux_tower_inst.lon
    point = Point(lon, lat)

    mgrs = _find_mgrs_tile(point, cfg["paths"]["sentinel_tile_shapefile"])
    logger.info("Using MGRS tile: %s", mgrs)

    buffer_days = s2_cfg["search_buffer_days"]
    t0 = (flux_tower_inst.flux_data["datetime_utc"].iloc[0] - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    t1 = (flux_tower_inst.flux_data["datetime_utc"].iloc[-1] + timedelta(days=buffer_days)).strftime("%Y-%m-%d")

    bbox = make_bbox(lat, lon, footprint_size)

    catalog = Client.open(s2_cfg["stac_endpoint"])
    search = catalog.search(
        collections=[s2_cfg["collection"]],
        datetime=f"{t0}/{t1}",
        query={
            "grid:code": {"eq": f"MGRS-{mgrs}"},
            "eo:cloud_cover": {"lt": s2_cfg["cloud_cover_max_pct"]},
        },
    )
    items = list(search.items())
    logger.info("Found %d Sentinel-2 items", len(items))
    if not items:
        raise RuntimeError("No Sentinel-2 items found for the requested site/time range.")

    bands = s2_cfg["bands"]
    stack = stackstac.stack(
        items, assets=bands, resolution=s2_cfg["resolution_m"], bounds_latlon=bbox, dtype="float64", rescale=False
    )
    cube = stack.to_dataset(dim="band")
    cube = cube.rio.write_crs(cube.rio.crs)

    # Reflectance bands are scaled 0-10000 in L2A; SCL is a class label and must not be scaled.
    bands_to_scale = [b for b in bands if b != "scl"]
    cube[bands_to_scale] /= 10000
    cube = cube.chunk({"time": -1})

    logger.info("Cube size: %.2f GB", cube.nbytes / 1e9)
    return cube, bbox, point


def mask_sentinel2(handler, vprm_inst, cfg):
    """Determine the dominant SCL class per pixel across time; mask indices where it's water."""
    s2_cfg = source_cfg(cfg)
    scl_codes = s2_cfg["scl_classes"]
    pool = [scl_codes[name] for name in s2_cfg["dominant_class_pool"]]
    water_code = scl_codes[s2_cfg["water_class"]]

    scl = handler.sat_img["scl"]
    scl_valid = scl.where(scl.isin(pool))
    counts = xr.concat([(scl_valid == c).sum(dim="time") for c in pool], dim="class")
    counts = counts.assign_coords({"class": pool})
    dominant = counts.idxmax(dim="class")

    vprm_inst.sat_imgs.sat_img["dominant_scl"] = dominant
    for sat_ind in s2_cfg["indices"]:
        vprm_inst.sat_imgs.sat_img[sat_ind] = vprm_inst.sat_imgs.sat_img[sat_ind].where(
            vprm_inst.sat_imgs.sat_img["dominant_scl"] != water_code
        )

    vprm_inst.calc_min_max_evi_lswi()


def rgb_quicklook_index(cube, cfg):
    """Timestep with the most pixels in the 'valid' class, as a rough cloud-free proxy."""
    s2_cfg = source_cfg(cfg)
    valid_code = s2_cfg["scl_classes"][s2_cfg["rgb_quicklook_valid_class"]]
    valid_count = (cube.scl == valid_code).sum(dim=("y", "x"))
    ind = int(valid_count.argmax(dim="time").values)
    logger.info("Quicklook timestep index %d (valid pixel count: %d)", ind, int(valid_count.max()))
    return ind


# ---------------------------------------------------------------------------
# MODIS (not yet implemented)
# ---------------------------------------------------------------------------

def fetch_modis_stack(flux_tower_inst, footprint_size, cfg):
    raise NotImplementedError(
        "MODIS support isn't implemented yet. You'll need: (1) a STAC search "
        "against a MODIS surface-reflectance/vegetation-index collection "
        "(e.g. Planetary Computer's modis-09A1-061 or modis-13Q1-061), "
        "(2) a pyVPRM.sat_managers.modis handler mirroring sentinel2's "
        "interface (crop(), sat_img, etc.), and (3) advanced.satellite.modis "
        "in config.yaml filled in (band_map, bands, indices - drop 'ndre', "
        "MODIS has no red-edge band)."
    )


def mask_modis(handler, vprm_inst, cfg):
    raise NotImplementedError(
        "MODIS quality masking isn't implemented yet - use the product's QA/"
        "state bands (e.g. MOD09 state_1km) instead of Sentinel-2's SCL."
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SOURCES = {
    "sentinel2": {"fetch": fetch_sentinel2_stack, "mask": mask_sentinel2},
    "modis": {"fetch": fetch_modis_stack, "mask": mask_modis},
}


def fetch_satellite_stack(flux_tower_inst, footprint_size, cfg):
    return SOURCES[cfg["satellite"]["source"]]["fetch"](flux_tower_inst, footprint_size, cfg)


def mask_satellite(handler, vprm_inst, cfg):
    return SOURCES[cfg["satellite"]["source"]]["mask"](handler, vprm_inst, cfg)
