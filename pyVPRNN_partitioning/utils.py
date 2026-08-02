"""Small standalone helpers shared by run_vprm_pipeline.py and train_and_evaluate.py."""

import math
import os
import sys
import time
import logging
import numpy as np
from pyproj import Transformer
logger = logging.getLogger("vprm_pipeline")


def build_fixed_footprint(ds_cropped, half_size=2):
    """
    A static footprint: uniform weight over a (2*half_size+1) x (2*half_size+1)
    pixel box centered on the tower's own grid cell - default half_size=2
    gives a 5x5 box. Normalized to sum to 1, matching the real time-varying
    FFP footprint's own normalization, so footprint_weighted_sum() results
    using this fixed footprint are on the same scale as - and directly
    comparable to - results using the real footprint.
    """
    site_lon = ds_cropped.attrs["site_lon"]
    site_lat = ds_cropped.attrs["site_lat"]

    transformer = Transformer.from_crs("EPSG:4326", ds_cropped.rio.crs, always_xy=True)
    site_x, site_y = transformer.transform(site_lon, site_lat)

    x_idx = int(np.argmin(np.abs(ds_cropped["x"].values - site_x)))
    y_idx = int(np.argmin(np.abs(ds_cropped["y"].values - site_y)))

    ny, nx = ds_cropped.sizes["y"], ds_cropped.sizes["x"]
    footprint = np.zeros((ny, nx), dtype=np.float32)

    y_lo, y_hi = max(0, y_idx - half_size), min(ny, y_idx + half_size + 1)
    x_lo, x_hi = max(0, x_idx - half_size), min(nx, x_idx + half_size + 1)
    if (y_hi - y_lo) < (2 * half_size + 1) or (x_hi - x_lo) < (2 * half_size + 1):
        logger.warning(
            "Fixed footprint box clipped by domain edge (tower pixel too close "
            "to the crop boundary) - got %dx%d instead of the requested %dx%d.",
            y_hi - y_lo, x_hi - x_lo, 2 * half_size + 1, 2 * half_size + 1,
        )

    footprint[y_lo:y_hi, x_lo:x_hi] = 1.0
    footprint /= footprint.sum()
    return footprint

def retry_with_backoff(fn, max_attempts=5, base_delay=5, description="operation"):
    """
    Retry fn() with exponential backoff (base_delay * 2**attempt seconds).

    Useful for transient upstream failures against external services - STAC
    catalog searches, tile downloads, Earthdata Hub reads - which have shown
    up repeatedly as one-off 5xx/gateway-timeout errors (Azure Front Door
    OriginTimeout, EDH 500s) rather than actual bugs. Catches broadly (any
    Exception) since the failure modes across pystac_client/requests/aiohttp
    aren't worth enumerating individually here - the final attempt always
    re-raises, so a genuine, non-transient bug still surfaces, just after
    max_attempts tries instead of immediately.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.error("%s failed after %d attempts: %s", description, max_attempts, e)
                raise
            wait = base_delay * (2 ** attempt)
            logger.warning(
                "%s failed (attempt %d/%d): %s - retrying in %ds...",
                description, attempt + 1, max_attempts, e, wait,
            )
            time.sleep(wait)


def maybe_extend_syspath(cfg):
    """
    Best-effort fallback for environments where pyVPRM isn't pip-installed:
    add paths.pyvprm_lib_path / paths.pyvprm_repo_path from config.yaml to
    sys.path if they're set and not already on it. Prefer
    `pip install -e /path/to/pyVPRM` over relying on this long-term - and
    if you do rely on it, this is the one place both scripts read it from,
    so a path change in config.yaml only needs to happen once.
    """
    for key in ("pyvprm_lib_path", "pyvprm_repo_path"):
        path = cfg.get("paths", {}).get(key)
        if path and path not in sys.path:
            sys.path.insert(0, path)


def site_base_path(output_base_dir, land_cover_type, site_id, suffix=""):
    """
    Single source of truth for the per-site output directory name, so the
    pipeline script (which creates it) and the training script (which reads
    from it) always agree. Historically this was hardcoded independently in
    both places and had drifted out of sync - always go through this
    function rather than rebuilding the path inline.
    """
    return os.path.join(output_base_dir, f"{land_cover_type}_{site_id}{suffix}")


def km_to_deg(lat_center, lon_center, km_lat, km_lon=None):
    """Convert a box size in km to approximate lat/lon bounds around a center point."""
    deg_per_km_lat = 1 / 111.32
    delta_lat = km_lat * deg_per_km_lat

    if km_lon is None:
        km_lon = km_lat

    deg_per_km_lon = 1 / (111.32 * math.cos(math.radians(lat_center)))
    delta_lon = km_lon * deg_per_km_lon

    return (
        [lat_center - delta_lat / 2, lat_center + delta_lat / 2],
        [lon_center - delta_lon / 2, lon_center + delta_lon / 2],
    )


def make_bbox(lat, lon, footprint_size_m):
    """Build a (minx, miny, maxx, maxy) bbox of the given half-size (in meters) around a point."""
    half_size_km = footprint_size_m / 1000
    lat_rad = np.deg2rad(lat)

    dlat = half_size_km / 111.32
    dlon = half_size_km / (111.32 * np.cos(lat_rad))

    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def plot_footprint_contours(ax, footprint, percentiles=(0.5, 0.9, 0.95), color="red", lw=1.2):
    """
    Overlay integrated footprint mass contours (e.g. 50/90/95%) on an existing axis.

    Parameters
    ----------
    ax : matplotlib axis
    footprint : xr.DataArray, 2D (y, x)
    percentiles : tuple of integrated mass fractions to contour
    """
    data = footprint.values

    weights_sorted = np.sort(data.flatten())[::-1]
    cumsum = np.cumsum(weights_sorted)
    cumsum /= cumsum[-1]

    levels = sorted(
        weights_sorted[np.searchsorted(cumsum, p)] for p in percentiles
    )

    ax.contour(footprint["x"], footprint["y"], data, levels=levels, colors=color, linewidths=lw)


def footprint_weighted_sum(field, footprint):
    """
    Collapse a per-sample spatial field (batch, H, W) to a per-sample scalar
    by weighting with the (batch, H, W) footprint and summing over space.
    Used to turn pixel-level GPP/Reco maps into a single footprint-integrated
    flux comparable to the tower's NEE_VUT_REF.
    """
    return np.sum(field * footprint, axis=1).sum(axis=1)


def geodesic_point_buffer(lat, lon, km):
    """A circle of the given radius (km), returned as a lon/lat coordinate ring."""
    from pyproj import Transformer
    from shapely.geometry import Point

    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    t = Transformer.from_crs(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0",
        "+proj=longlat +datum=WGS84",
    )
    return [t.transform(x, y) for x, y in buf.exterior.coords[:]]


def crop_dataset_by_radius(ds, lonlat, radius_km):
    """Crop a CRS-tagged, rio-enabled xr.Dataset to a geodesic circle around (lon, lat)."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    lon, lat = lonlat
    circle_poly = gpd.GeoSeries(Polygon(geodesic_point_buffer(lat, lon, radius_km)), crs="EPSG:4326")
    if circle_poly.crs != ds.rio.crs:
        circle_poly = circle_poly.to_crs(ds.rio.crs)

    minx, miny, maxx, maxy = circle_poly.total_bounds
    return ds.sel(x=slice(minx, maxx), y=slice(maxy, miny))


# ---------------------------------------------------------------------------
# VPRM config / land-cover-class resolution
# ---------------------------------------------------------------------------

def resolve_vprm_config_path(cfg):
    """
    Path to the VPRM vegetation-class/temperature-parameter yaml (the one
    with vprm_class/tmin/topt/tmax/tlow per land-cover type). If
    model.vprm_config_path is set in config.yaml, use that; otherwise fall
    back to pyVPRM's own built-in default at <pyVPRM install>/vprm_configs/
    pyvprnn.yaml.

    run_vprm_pipeline.py uses this file to build land_cover_map in the
    first place; train_and_evaluate.py uses it to size and label the
    model's static land-cover input. Both must resolve to the same file or
    they'll silently disagree about how many/which classes exist - always
    go through this function rather than hardcoding the path independently.

    Note this is a different file from copernicus_land_cover.yaml (which
    maps Copernicus's own land-cover codes onto this scheme) - don't
    conflate the two.
    """
    configured = cfg.get("model", {}).get("vprm_config_path")
    if configured:
        return configured
    import pyVPRM
    return os.path.join(pyVPRM.__path__[0], "vprm_configs", "pyvprnn.yaml")


def load_vprm_land_cover_classes(vprm_config_path):
    """The vprm_class integer codes defined in a VPRM config yaml, sorted ascending."""
    import yaml
    with open(vprm_config_path) as f:
        veg_cfg = yaml.safe_load(f)
    return sorted(entry["vprm_class"] for entry in veg_cfg.values())


def build_pyvprnn_kwargs(cfg, model_name=None):
    """
    Build the constructor kwargs for pyvprnn_v1/pyvprnn_v2 that need to stay
    config-driven, and only those: land_cover_classes (derived from
    vprm_config_path, so it can't silently disagree with whatever built
    land_cover_map in run_vprm_pipeline.py) and, for pyvprnn_v2, the lag
    settings.

    sat_vars/met_vars/gpp_met_vars/reco_met_vars/met_scaling are
    deliberately NOT included here - they're left as pyvprnn_v1/v2's own
    hardcoded DEFAULT_* class attributes. Making everything config-driven
    added more moving parts than it was worth; land_cover_classes is the
    one exception because getting it wrong silently breaks the static
    input's shape/meaning, not just a modeling choice.

    model_name selects whether the v2-only lag kwargs (lagged_met_vars,
    variable_lags, lag_window) are included - pyvprnn_v1's constructor
    doesn't accept them at all, so they must NOT be passed when
    instantiating v1. Defaults to cfg["model"]["version"] if not given.
    """
    model_cfg = cfg["model"]
    vprm_config_path = resolve_vprm_config_path(cfg)
    kwargs = {
        "land_cover_classes": load_vprm_land_cover_classes(vprm_config_path),
    }

    model_name = model_name or cfg["model"]["version"]
    if model_name == "pyvprnn_v2":
        lag_cfg = model_cfg.get("lag", {})
        kwargs.update({
            "lagged_met_vars": lag_cfg.get("lagged_met_vars"),
            "variable_lags": lag_cfg.get("variable_lags"),
            "lag_window": lag_cfg.get("lag_window"),
        })

    return kwargs


def resolve_pyvprnn_model(model_name):
    """
    Return (model_class, generator_class, is_lagged) for a given pyvprnn
    version name. Centralizes the v1/v2 dispatch so every call site in
    train_and_evaluate.py (dataset loading, inference, PDP) agrees on what
    each model name means, rather than each re-implementing its own
    if/elif independently - which had already drifted out of sync (one
    branch had a syntax error, another silently defaulted to v1).
    """
    from pyVPRM.vprm_models.pyvprnn_v1 import pyvprnn_v1, BatchGenerator

    if model_name == "pyvprnn_v1":
        return pyvprnn_v1, BatchGenerator, False
    if model_name == "pyvprnn_v2":
        from pyVPRM.vprm_models.pyvprnn_v2 import pyvprnn_v2, LaggedBatchGenerator
        return pyvprnn_v2, LaggedBatchGenerator, True
    raise ValueError(f"Unknown model '{model_name}' - expected 'pyvprnn_v1' or 'pyvprnn_v2'.")


def make_generator(pyvprnn_inst, is_lagged, **kwargs):
    """
    Construct the right BatchGenerator/LaggedBatchGenerator for an already-
    instantiated pyvprnn_inst, pulling land_cover_classes (and, if lagged,
    the lag settings) off the instance itself rather than needing them
    passed in separately.
    """
    _, generator_cls, _ = resolve_pyvprnn_model("pyvprnn_v2" if is_lagged else "pyvprnn_v1")
    if is_lagged:
        return generator_cls(
            pyvprnn_inst.ds_cropped, pyvprnn_inst.sat_vars, pyvprnn_inst.met_vars,
            land_cover_classes=pyvprnn_inst.land_cover_classes,
            lagged_met_vars=pyvprnn_inst.lagged_met_vars,
            variable_lags=pyvprnn_inst.variable_lags,
            lag_window=pyvprnn_inst.lag_window,
            **kwargs,
        )
    return generator_cls(
        pyvprnn_inst.ds_cropped, pyvprnn_inst.sat_vars, pyvprnn_inst.met_vars,
        land_cover_classes=pyvprnn_inst.land_cover_classes,
        **kwargs,
    )