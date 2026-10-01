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
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import dask
import numpy as np
import pandas as pd
import requests
import xarray as xr
import geopandas as gpd
import stackstac
from pystac_client import Client
from pyproj import Transformer
from shapely.geometry import Point
from utils import make_bbox
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


def _search_window(flux_tower_inst, buffer_days):
    """Tower record extended by `buffer_days` on both sides, as ISO date strings."""
    t0 = (flux_tower_inst.flux_data["datetime_utc"].iloc[0]
          - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    t1 = (flux_tower_inst.flux_data["datetime_utc"].iloc[-1]
          + timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    return t0, t1


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

    t0, t1 = _search_window(flux_tower_inst, s2_cfg["search_buffer_days"])
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


# ---------------------------------------------------------------------------
# RGB quicklook (source-aware)
# ---------------------------------------------------------------------------

def rgb_quicklook_index(cube, cfg):
    """
    Timestep with the most 'clear' pixels, as a rough cloud-free proxy.

    Sentinel-2 uses the SCL class named in rgb_quicklook_valid_class; HLS has no
    categorical class layer, so clear means "no QA bit set" in Fmask (bits 0-5:
    cirrus, cloud, adjacent cloud/shadow, shadow, snow/ice, water - the aerosol
    bits 6-7 are deliberately ignored).
    """
    source = cfg["satellite"]["source"]
    src = source_cfg(cfg)

    if source == "hls":
        fmask = cube[src["quality_band"]]
        clear = (fmask.fillna(255).astype("int32") & 0b0011_1111) == 0
        valid_count = clear.sum(dim=("y", "x"))
    else:
        valid_code = src["scl_classes"][src["rgb_quicklook_valid_class"]]
        valid_count = (cube[src.get("quality_band", "scl")] == valid_code).sum(dim=("y", "x"))

    ind = int(valid_count.argmax(dim="time").values)
    logger.info("Quicklook timestep index %d (valid pixel count: %d)",
                ind, int(valid_count.max()))
    return ind


# ---------------------------------------------------------------------------
# HLS (Harmonized Landsat Sentinel-2) - NASA Earthdata / LP DAAC
# ---------------------------------------------------------------------------
# Source is LP DAAC's CMR STAC, not Planetary Computer: PC's hls2-* mirror only
# starts in 2020, while LP DAAC holds the full v2.0 archive (L30 from
# 2013-04-11, S30 from 2015-11-28).
#
# Three things make this more involved than the Sentinel-2 path above:
#
#  1. Auth. Assets sit behind Earthdata Login. LP DAAC answers a data request
#     with a redirect to a short-lived pre-signed CloudFront URL, and GDAL
#     cannot follow that reliably - it caches HEAD-signed redirects and reuses
#     them for GET (403), and an Authorization header sent along to the signed
#     URL is rejected as well. So the redirects are resolved here with requests
#     (bearer token, allow_redirects=False) and stackstac only ever sees plain
#     signed URLs needing no auth.
#  2. Expiry. Those signed URLs are valid ~1 h, so the date range is walked in
#     chunks (advanced.satellite.hls.chunk_freq) that are searched, signed and
#     read back-to-back, then concatenated.
#  3. Band names. L30 and S30 use different band codes for the same physical
#     band, and CMR STAC exposes raw codes only (no common-name aliases), so
#     assets are renamed before stacking.
#
# Requires an Earthdata token in EARTHDATA_TOKEN (Profile -> Generate Token at
# https://urs.earthdata.nasa.gov). Tokens expire after 60 days.

_HLS_CMR_STAC = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"

# First acquisition in each HLS v2.0 collection.
_HLS_ARCHIVE_START = {"HLSL30_2.0": "2013-04-11", "HLSS30_2.0": "2015-11-28"}

# Raw LP DAAC asset keys -> common names. "nir" deliberately maps to S30's B8A
# (narrow NIR), not B08: B8A is the band HLS bandpass-adjusts to OLI, so it is
# the harmonized counterpart of L30's B05. L30 has no red-edge bands at all,
# which is why rededge* (and the ndre index) cannot be used in a mixed stack.
_HLS_BAND_CROSSWALK = {
    "L30": {
        "B01": "coastal", "B02": "blue", "B03": "green", "B04": "red",
        "B05": "nir", "B06": "swir16", "B07": "swir22",
        "B09": "cirrus", "B10": "lwir11", "B11": "lwir12", "Fmask": "Fmask",
    },
    "S30": {
        "B01": "coastal", "B02": "blue", "B03": "green", "B04": "red",
        "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
        "B08": "nir08_broad", "B8A": "nir", "B09": "wvp", "B10": "cirrus",
        "B11": "swir16", "B12": "swir22", "Fmask": "Fmask",
    },
}

# Reflectance is stored as int16 with scale 0.0001 and fill -9999; Fmask is a
# bit-packed QA band and must not be scaled.
_HLS_REFLECTANCE_BANDS = {
    "coastal", "blue", "green", "red", "nir", "nir08_broad",
    "rededge1", "rededge2", "rededge3", "swir16", "swir22", "cirrus",
}
_HLS_SCALE = np.float32(1e-4)
_HLS_FILL = -9999

# GDAL settings for reading the pre-signed URLs. CPL_VSIL_CURL_USE_HEAD="NO"
# is the important one: with HEAD enabled GDAL caches a HEAD-signed redirect
# and then reuses it for the range GETs, which CloudFront rejects with 403.
_HLS_GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_USE_HEAD="NO",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
)

_hls_thread_local = threading.local()


def _hls_token():
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "EARTHDATA_TOKEN is not set - HLS from LP DAAC needs an Earthdata "
            "Login token (https://urs.earthdata.nasa.gov -> Generate Token). "
            "Tokens expire after 60 days."
        )
    return token


def _hls_product(item):
    key = f"{item.collection_id or ''} {item.id}"
    if "L30" in key:
        return "L30"
    if "S30" in key:
        return "S30"
    raise ValueError(f"Cannot determine HLS product for item {item.id}")


def _harmonize_hls_band_names(items):
    """Rename each item's assets in place from raw band codes to shared common names."""
    for item in items:
        xwalk = _HLS_BAND_CROSSWALK[_hls_product(item)]
        item.assets = {xwalk.get(k, k): v for k, v in item.assets.items()}
    return items


def _hls_date_chunks(start_date, end_date, freq="YS"):
    """Split [start, end] into non-overlapping inclusive periods."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if end < start:
        return []
    edges = [start] + [t for t in pd.date_range(start, end, freq=freq) if t > start]
    periods = []
    for i, s in enumerate(edges):
        e = edges[i + 1] - pd.Timedelta(days=1) if i + 1 < len(edges) else end
        periods.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    return periods


def _hls_utm_epsg(bbox):
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def _hls_snapped_bounds(bbox, epsg, resolution):
    """
    Fixed bounds in the target CRS, snapped outward to the pixel grid.

    Given bounds_latlon, stackstac derives the grid from the items it was
    handed, so the x/y size can differ by a pixel between chunks and the
    concat below would fail. Identical, grid-snapped bounds for every chunk
    avoid that.
    """
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    pts = [t.transform(x, y) for x in (bbox[0], bbox[2]) for y in (bbox[1], bbox[3])]
    xs, ys = zip(*pts)
    return (float(np.floor(min(xs) / resolution) * resolution),
            float(np.floor(min(ys) / resolution) * resolution),
            float(np.ceil(max(xs) / resolution) * resolution),
            float(np.ceil(max(ys) / resolution) * resolution))


def _search_hls_items(bbox, start_date, end_date, collections, max_cloud_cover=None):
    """CMR STAC search. CMR has no `query` extension, so cloud filtering is client-side."""
    catalog = Client.open(_HLS_CMR_STAC)
    search = catalog.search(
        collections=list(collections),
        bbox=list(bbox),
        datetime=f"{start_date}/{end_date}",
        limit=100,   # page size; CMR STAC is slow with large pages
    )
    items = list(search.items())

    if max_cloud_cover is not None:
        items = [i for i in items
                 if i.properties.get("eo:cloud_cover") is None
                 or i.properties["eo:cloud_cover"] < max_cloud_cover]

    items = list({i.id: i for i in items}.values())   # CMR can return duplicates
    items.sort(key=lambda i: i.datetime)
    return _harmonize_hls_band_names(items)


def _hls_session():
    if not hasattr(_hls_thread_local, "session"):
        _hls_thread_local.session = requests.Session()
    return _hls_thread_local.session


def _resolve_signed_url(href, token, retries=3):
    last = None
    for _ in range(retries):
        try:
            r = _hls_session().get(href, headers={"Authorization": f"Bearer {token}"},
                                   allow_redirects=False, stream=True, timeout=60)
            r.close()
            if r.status_code in (301, 302, 303, 307, 308):
                return r.headers["location"]
            last = f"HTTP {r.status_code}"
            if r.status_code in (401, 403):
                break  # token problem, retrying won't help
        except requests.RequestException as e:
            last = repr(e)
    raise RuntimeError(f"Could not sign {href}: {last}")


def _presign_hls_items(items, bands, token, max_workers=8):
    """
    Replace asset hrefs with pre-signed CloudFront URLs (valid ~1 h).
    Assets that fail to sign are dropped and reported, so one broken granule
    does not abort a multi-year run.
    """
    jobs = [(item, b) for item in items for b in bands if b in item.assets]

    def work(job):
        item, b = job
        try:
            return job, _resolve_signed_url(item.assets[b].href, token), None
        except RuntimeError as e:
            return job, None, e

    failed = []
    with ThreadPoolExecutor(max_workers) as ex:
        for (item, b), url, err in ex.map(work, jobs):
            if url is not None:
                item.assets[b].href = url
            else:
                failed.append((item.id, b, str(err)))
                del item.assets[b]

    if any("HTTP 401" in f[2] or "HTTP 403" in f[2] for f in failed):
        raise RuntimeError(f"Earthdata authorization failed (check token): {failed[0][2]}")
    for fid, b, err in failed:
        logger.warning("HLS asset skipped: %s %s (%s)", fid, b, err)

    # Keep items that still have the quality band and at least one other band;
    # bands a product simply doesn't have are filled with NaN by stackstac.
    def usable(i):
        if "Fmask" in bands and "Fmask" not in i.assets:
            return False
        return any(b in i.assets for b in bands if b != "Fmask")

    return [i for i in items if usable(i)]


def _load_hls_chunk(items, bounds, bands, epsg, resolution, read_workers):
    """Read one set of already-signed items into memory. `bounds` is in EPSG:<epsg>."""
    from rasterio.errors import RasterioIOError

    stack = stackstac.stack(
        items,
        assets=list(bands),
        bounds=tuple(bounds),
        epsg=epsg,
        resolution=resolution,
        dtype="float32",
        fill_value=np.float32("nan"),
        # rescale=True trips a numpy safe-casting error against HLS's integer
        # scale/offset metadata (a stackstac quirk) - scaled explicitly below.
        rescale=False,
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(always=_HLS_GDAL_ENV),
        # A granule listed in CMR but missing from the archive shouldn't kill
        # the run; that scene becomes NaN instead.
        errors_as_nodata=(RasterioIOError("HTTP response code: 404"),),
    )

    # Keep only light-weight coordinates; the rest are per-item STAC properties
    # that would otherwise ride along through every later reduction.
    keep = {"time", "x", "y", "band", "id"}
    stack = stack.drop_vars([c for c in stack.coords if c not in keep])
    ids = [str(v) for v in np.atleast_1d(stack["id"].values)]
    stack = stack.assign_coords(
        sensor=("time", ["landsat" if ".L30." in i else "sentinel" for i in ids])
    )
    stack.attrs = {}

    cube = stack.to_dataset(dim="band")
    for b in cube.data_vars:
        if b in _HLS_REFLECTANCE_BANDS:
            cube[b] = cube[b].where(cube[b] != _HLS_FILL) * _HLS_SCALE

    # Threads, not processes: these are HTTP range reads, and GDAL releases the
    # GIL while waiting on them.
    with dask.config.set(scheduler="threads", num_workers=read_workers):
        return cube.compute()


def fetch_hls_stack(flux_tower_inst, footprint_size, cfg):
    hls_cfg = source_cfg(cfg)
    lat, lon = flux_tower_inst.lat, flux_tower_inst.lon
    point = Point(lon, lat)

    token = _hls_token()
    collections = list(hls_cfg["collections"])
    bands = list(hls_cfg["bands"])
    resolution = hls_cfg["resolution_m"]
    max_cloud_cover = hls_cfg["cloud_cover_max_pct"]
    sign_workers = hls_cfg.get("sign_workers", 8)
    read_workers = hls_cfg.get("read_workers", 8)

    t0, t1 = _search_window(flux_tower_inst, hls_cfg["search_buffer_days"])

    # Nothing exists before the first HLS acquisition; asking for it only adds
    # empty chunks and a search round-trip each.
    archive_start = min(_HLS_ARCHIVE_START[c] for c in collections)
    if t0 < archive_start:
        logger.info("Clamping HLS search start %s -> %s (archive start)", t0, archive_start)
        t0 = archive_start

    bbox = make_bbox(lat, lon, footprint_size)
    epsg = _hls_utm_epsg(bbox)
    bounds = _hls_snapped_bounds(bbox, epsg, resolution)

    cubes = []
    for start, end in _hls_date_chunks(t0, t1, hls_cfg.get("chunk_freq", "YS")):
        items = _search_hls_items(bbox, start, end, collections, max_cloud_cover)
        if not items:
            logger.info("HLS %s..%s: no items", start, end)
            continue
        items = _presign_hls_items(items, bands, token, sign_workers)
        if not items:
            logger.warning("HLS %s..%s: no usable items after signing", start, end)
            continue
        cube = _load_hls_chunk(items, bounds, bands, epsg, resolution, read_workers)
        logger.info("HLS %s..%s: %d scenes", start, end, cube.sizes["time"])
        cubes.append(cube)

    if not cubes:
        raise RuntimeError("No HLS items found for the requested site/time range.")

    cube = xr.concat(cubes, dim="time", coords="minimal", compat="override",
                     join="exact", combine_attrs="drop_conflicts").sortby("time")

    sensors = [str(s) for s in np.atleast_1d(cube["sensor"].values)]
    logger.info("Found %d HLS scenes (%d L30 + %d S30), %s to %s",
                cube.sizes["time"], sensors.count("landsat"), sensors.count("sentinel"),
                str(cube["time"].values[0])[:10], str(cube["time"].values[-1])[:10])

    cube = cube.drop_vars([c for c in ("id", "sensor") if c in cube.coords])
    cube = cube.rio.write_crs(f"EPSG:{epsg}")
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