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
import planetary_computer
from utils import make_bbox
import sys
from stackstac.rio_reader import DEFAULT_GDAL_ENV

logger = logging.getLogger("vprm_pipeline")

def ensure_pyvprm_importable(cfg):
    """
    Make sure `import pyVPRM` (and pyVPRM.sat_managers.*) will succeed.
 
    If pyVPRM is pip-installed, this is a no-op. If it isn't, fall back to
    cfg["paths"]["pyvprm_repo_path"] - the path to a local pyVPRM checkout -
    added to sys.path so it can be imported like a normal package.
    """
    try:
        import pyVPRM  # noqa: F401
        return
    except ImportError:
        pass
 
    repo_path = cfg.get("paths", {}).get("pyvprm_repo_path")
    if not repo_path:
        raise ImportError(
            "pyVPRM isn't installed and no paths.pyvprm_repo_path is set in "
            "config.yaml to fall back to a local checkout."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
 
    import pyVPRM  # noqa: F401  - retry now that repo_path is on sys.path



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


# ---------------------------------------------------------------------------
# Sentinel-2 L2A radiometric scaling
# ---------------------------------------------------------------------------
# L2A products are distributed as scaled integers. Until processing baseline
# 04.00 (2022-01-25) the conversion was simply DN / 10000. From 04.00 onward
# ESA added a BOA_ADD_OFFSET of -1000 DN (= -0.1 reflectance), so the correct
# conversion became (DN - 1000) / 10000. Applying the old formula to new-
# baseline data biases every reflectance high by 0.1, which does not cancel in
# ratio indices like EVI/LSWI and is therefore silently wrong rather than
# merely offset. Baseline can vary item-to-item within one search (ESA
# reprocesses), so scale/offset are resolved per item, preferring the
# per-asset raster:bands metadata and falling back to the baseline number.

_S2_DEFAULT_SCALE = 1e-4
_S2_BOA_OFFSET_BASELINE = 4.0


def _s2_band_scale_offset(item, band):
    """(scale, offset) in reflectance units for one asset of one item."""
    asset = item.assets.get(band)
    if asset is not None:
        raster_bands = asset.extra_fields.get("raster:bands") or []
        meta = raster_bands[0] if raster_bands else {}
        scale, offset = meta.get("scale"), meta.get("offset")
        if scale is not None and offset is not None:
            return float(scale), float(offset)

    try:
        baseline = float(item.properties.get("s2:processing_baseline", "0"))
    except (TypeError, ValueError):
        baseline = 0.0
    offset = -0.1 if baseline >= _S2_BOA_OFFSET_BASELINE else 0.0
    return _S2_DEFAULT_SCALE, offset


def _apply_s2_boa_scaling(cube, items, bands, quality_band="scl"):
    """Convert DN to surface reflectance per item, honouring the BOA offset.

    stackstac sorts the stack by date, so the cube's time axis is not in
    `items` order - the mapping is done via the per-timestep "id" coordinate
    rather than positionally.
    """
    by_id = {it.id: it for it in items}
    ids = [str(i) for i in cube["id"].values]

    missing = [i for i in ids if i not in by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(ids)} cube timesteps have no matching STAC "
            f"item (first: {missing[0]}) - cannot determine BOA scaling."
        )

    baselines = sorted({by_id[i].properties.get("s2:processing_baseline") or "unknown"
                        for i in ids})
    logger.info("Sentinel-2 processing baselines in stack: %s", ", ".join(baselines))

    offsets_seen = set()
    for band in bands:
        if band == quality_band:
            continue
        pairs = [_s2_band_scale_offset(by_id[i], band) for i in ids]
        offsets_seen.update(p[1] for p in pairs)
        # Deliberately no time coordinate: broadcasting is positional along the
        # "time" dim, so duplicate timestamps can't trigger an xarray align.
        scale = xr.DataArray(np.array([p[0] for p in pairs]), dims="time")
        offset = xr.DataArray(np.array([p[1] for p in pairs]), dims="time")
        cube[band] = cube[band] * scale + offset

    logger.info("Applied BOA offsets: %s", sorted(offsets_seen))
    return cube

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

    # Assets live in a public AWS Open Data bucket; without the anonymous flag
    # GDAL looks for credentials and fails. stackstac opens datasets inside its
    # own rasterio.Env, so process-level env vars are not reliably inherited.
    gdal_env = DEFAULT_GDAL_ENV.updated(always=dict(
        AWS_NO_SIGN_REQUEST="YES",
        AWS_DEFAULT_REGION="us-west-2",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    ))

    bands = s2_cfg["bands"]
    quality_band = s2_cfg.get("quality_band", "scl")
    stack = stackstac.stack(
        items,
        assets=bands,
        resolution=s2_cfg["resolution_m"],
        bounds_latlon=bbox,
        dtype="float64",
        fill_value=np.nan,
        # Scaling is applied explicitly below rather than by stackstac, so the
        # per-item baseline/offset actually used gets logged and unmapped items
        # fail loudly instead of being silently mis-scaled.
        rescale=False,
        gdal_env=gdal_env,
    )
    cube = stack.to_dataset(dim="band")
    cube = cube.rio.write_crs(cube.rio.crs)

    # SCL is a categorical class label and must not be scaled.
    cube = _apply_s2_boa_scaling(cube, items, bands, quality_band=quality_band)
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
# HLS (Harmonized Landsat Sentinel-2)
# ---------------------------------------------------------------------------
# Unlike earth-search's Sentinel-2 collection (which already exposes common
# band names like "red"/"nir08"), Planetary Computer's raw HLS assets are
# keyed by band code (B01, B04, ...), and the L30 (Landsat) and S30
# (Sentinel-2) collections use *different* codes for the same physical band
# (e.g. NIR is B05 on L30 but B08 on S30). We rename each item's assets to a
# shared set of common names before stacking, so both collections stack
# together cleanly and advanced.satellite.hls.band_map can reference plain
# names ("nir", "swir1", ...) the same way the Sentinel-2 block does.
_HLS_BAND_CROSSWALK = {
    "hls2-l30": {
        "B01": "coastal",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B05": "nir",
        "B06": "swir1",
        "B07": "swir2",
        "B09": "cirrus",
        "B10": "thermal1",
        "B11": "thermal2",
        "Fmask": "Fmask",
    },
    "hls2-s30": {
        "B01": "coastal",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B05": "rededge1",
        "B06": "rededge2",
        "B07": "rededge3",
        "B08": "nir",         # NIR-broad: shared with L30's B05, not bandpass-adjusted
        "B8A": "nir_narrow",  # Sentinel-only, no Landsat equivalent
        "B09": "watervapor",
        "B10": "cirrus",
        "B11": "swir1",
        "B12": "swir2",
        "Fmask": "Fmask",
    },
}
 
 
def _harmonize_hls_band_names(items):
    """Rename each item's assets in place from raw band codes to shared common names."""
    for item in items:
        crosswalk = _HLS_BAND_CROSSWALK.get(item.collection_id)
        if crosswalk is None:
            continue
        for raw_key, common_name in crosswalk.items():
            if raw_key in item.assets and raw_key != common_name:
                item.assets[common_name] = item.assets.pop(raw_key)
    return items
 
def fetch_hls_stack(flux_tower_inst, footprint_size, cfg):
    hls_cfg = source_cfg(cfg)
    lat, lon = flux_tower_inst.lat, flux_tower_inst.lon
    point = Point(lon, lat)
 
    buffer_days = hls_cfg["search_buffer_days"]
    t0 = (flux_tower_inst.flux_data["datetime_utc"].iloc[0] - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    t1 = (flux_tower_inst.flux_data["datetime_utc"].iloc[-1] + timedelta(days=buffer_days)).strftime("%Y-%m-%d")
 
    bbox = make_bbox(lat, lon, footprint_size)
 
    catalog = Client.open(hls_cfg["stac_endpoint"], modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=hls_cfg["collections"],
        bbox=bbox,
        datetime=f"{t0}/{t1}",
        query={"eo:cloud_cover": {"lt": hls_cfg["cloud_cover_max_pct"]}},
    )
    items = list(search.items())
    logger.info("Found %d HLS items (L30+S30)", len(items))
    if not items:
        raise RuntimeError("No HLS items found for the requested site/time range.")
 
    _harmonize_hls_band_names(items)
 
    bands = hls_cfg["bands"]
    # epsg must be pinned explicitly: bbox searches (unlike the Sentinel-2
    # single-MGRS-tile search above) can return items from adjacent UTM
    # zones near tile boundaries, and stackstac needs one common CRS to
    # stack into. L30/S30 share NASA's MGRS grid, so the first item's zone
    # is the right choice for a small footprint around one tower.
    epsg = items[0].properties.get("proj:epsg")
    stack = stackstac.stack(
        items,
        assets=bands,
        epsg=epsg,
        resolution=hls_cfg["resolution_m"],
        bounds_latlon=bbox,
        dtype="float32",
        fill_value=np.float32("nan"),
        # rescale=True trips a numpy safe-casting error against HLS's
        # integer scale/offset metadata (a stackstac quirk) - we scale to
        # reflectance ourselves below instead.
        rescale=False,
    )
    cube = stack.to_dataset(dim="band")
    cube = cube.rio.write_crs(cube.rio.crs)
    cube = cube.compute() 
 
    # HLS reflectance bands are scaled integers (scale factor 0.0001, per
    # the LP DAAC product spec); Fmask is a bit-packed QA band and must not
    # be scaled.
    quality_band = hls_cfg["quality_band"]
    bands_to_scale = [b for b in bands if b != quality_band]
    cube[bands_to_scale] *= hls_cfg["reflectance_scale_factor"]
    cube = cube.chunk({"time": -1})
 
    logger.info("Cube size: %.2f GB", cube.nbytes / 1e9)
    return cube, bbox, point
 
 
def mask_hls(handler, vprm_inst, cfg):
    """
    Permanent water-body exclusion, run post-Kalman - mirrors mask_sentinel2's
    dominant_scl logic exactly (both reduce over "time" before ever touching
    vprm_inst.sat_imgs.sat_img, so the resulting mask is (y, x)-only and
    broadcasts safely regardless of what vprm_inst's current temporal
    dimension is called - "time" pre-Kalman, "time_gap_filled" post-Kalman).
 
    Per-timestep QA masking (cloud/adjacent/shadow/snow) does NOT belong
    here: it already happened earlier and correctly, on handler.sat_img's
    original per-scene time axis, via add_sat_img(mask_bad_pixels=True,
    mask_clouds=True, mask_snow=True, mask_water=True, ...), which calls
    handler.mask_bad_pixels()/.mask_clouds()/.mask_snow()/.mask_water()
    directly. Redoing it here against handler.sat_img["Fmask"] would compare
    its original "time" dimension against vprm_inst.sat_imgs.sat_img's
    "time_gap_filled" dimension post-Kalman - two different, non-aligned
    xarray dimensions - and instead of erroring, xarray silently broadcasts
    across both, producing a spurious 4D result carrying both axes at once.
 
    Fmask's water bit is a per-scene spectral detection, not a stable
    land-cover class like SCL's, so unlike mask_sentinel2 (which reads a
    single categorical class straight off "scl"), this takes a majority
    vote across handler.sat_img's original time axis first to decide which
    pixels are permanent water.
    """
    is_water = handler.water_mask()
    dominant_water = is_water.sum(dim="time") > (is_water.sizes["time"] / 2)
 
    vprm_inst.sat_imgs.sat_img["dominant_water"] = dominant_water
    for sat_ind in source_cfg(cfg)["indices"]:
        vprm_inst.sat_imgs.sat_img[sat_ind] = vprm_inst.sat_imgs.sat_img[sat_ind].where(
            ~vprm_inst.sat_imgs.sat_img["dominant_water"]
        )
 
    vprm_inst.calc_min_max_evi_lswi()

 

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
    "hls": {"fetch": fetch_hls_stack, "mask": mask_hls},
    "modis": {"fetch": fetch_modis_stack, "mask": mask_modis},
}

 
def get_satellite_handler(cube, cfg):
    """Construct the right sat_manager instance (sentinel2/hls/...) for cfg's selected source."""
    ensure_pyvprm_importable(cfg)  # patches sys.path from paths.pyvprm_repo_path if needed
 
    from pyVPRM.sat_managers.sentinel2 import sentinel2
    from pyVPRM.sat_managers.hls import hls
 
    handler_classes = {
        "sentinel2": sentinel2,
        "hls": hls,
        # "modis": modis,  # add once pyVPRM.sat_managers.modis exists
    }
 
    source = cfg["satellite"]["source"]
    handler_cls = handler_classes.get(source)
    if handler_cls is None:
        raise NotImplementedError(
            f"No sat_manager handler registered for source '{source}' in get_satellite_handler()."
        )
    return handler_cls(sat_img=cube)

    
def fetch_satellite_stack(flux_tower_inst, footprint_size, cfg):
    return SOURCES[cfg["satellite"]["source"]]["fetch"](flux_tower_inst, footprint_size, cfg)


def mask_satellite(handler, vprm_inst, cfg):
    return SOURCES[cfg["satellite"]["source"]]["mask"](handler, vprm_inst, cfg)
