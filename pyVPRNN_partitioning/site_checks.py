"""
Unit and consistency checks for the site data (tower, ERA5, satellite, flux targets).

One table (VAR_SPECS) holds, per variable: expected unit, the plausible range of its monthly
median, the physically possible range, and the known unit errors with their conversion. Checks:

  1. Unit per MONTH (not per record): a month whose median lies outside the plausible range but
     inside it after one of the known conversions is converted. The same conversion for every
     month = record in the wrong unit; for some months only = unit change within the record
     (warning). Months that fit neither are flagged, not changed.
  2. Physically impossible values -> NaN (error if they are more than MAX_IMPOSSIBLE_SHARE).
  3. Constant runs (>= 6 h of identical non-zero values) in variables that are never constant
     (fill values) -> NaN; filled from the FLUXNET *_ERA counterpart where available.
  4. VPD against the saturation vapour pressure of TA, per month and per day (clean_vpd).
  5. Cross-checks: time stamps in UTC (solar noon from SW_IN_POT and from SW_IN against the site
     longitude), SW_IN <= potential radiation, no SW at night, tower vs ERA5 air temperature,
     soil vs air temperature, energy balance closure, NEE sign day / night, LE_RANDUNC scale,
     remaining -9999 sentinels, time axis regular, land-cover sums.

Unambiguous problems are corrected (the original is kept as <var>_orig, the correction is stored
in the variable's attributes); everything else is only reported. Works on the final xarray
dataset (check_dataset) and on the FLUXNET tower DataFrame of the pipeline (check_dataframe).

Every finding is a dict(level=info|correction|warning|error, var=..., msg=..., **details).
"""

import numpy as np
import pandas as pd
import xarray as xr

TIME_DIM = "datetime_utc"
MAX_IMPOSSIBLE_SHARE = 0.02      # more impossible values than this: error (no silent masking)
FLAT_STEPS = 12                  # 6 h of half-hourly data
# Masked tower values (constant runs, impossible VPD) are filled from the FLUXNET *_ERA counterpart
# (TA_ERA, VPD_ERA, SW_IN_ERA, ...; the same source FLUXNET uses for its own gap filling). False: they
# stay NaN and the half-hours drop out of training.
FILL_FROM_ERA = True
# Network inputs on a grid (satellite indices, ERA5 fields) must never become NaN: one NaN pixel makes
# the footprint sum, the loss and the gradients NaN. Impossible values there are replaced by linear
# interpolation in time of the same pixel (nearest valid value at the ends), clipped to the range.
# The same for the tower inputs of the network (after the ERA fill, if no ERA value exists).
TOWER_INPUTS = ("t2m", "TA_F", "ssrd", "SW_IN_F", "VPD_F", "P_F")
NO_NAN_VARS = ("nirv", "lswi", "t2m_era5", "stl1_era5", "stl2_era5", "swvl1_era5", "swvl2_era5",
               "sd_era5") + TOWER_INPUTS
# Satellite indices come from the gap filling / smoothing already: values outside the range are only
# clipped to it, never interpolated or masked.
CLIP_VARS = ("nirv", "lswi")


def _interp_array_along(da, dim):
    """Linear interpolation of NaNs along `dim` for every other index (pixel), nearest valid value at
    the ends; pandas only (no bottleneck). Pixels without any valid value stay NaN."""
    arr = np.moveaxis(np.asarray(da.values, dtype=np.float64), da.get_axis_num(dim), 0)
    shape = arr.shape
    flat = pd.DataFrame(arr.reshape(shape[0], -1))
    flat = flat.interpolate(method="linear", axis=0, limit_direction="both")
    out = np.moveaxis(flat.values.reshape(shape), 0, da.get_axis_num(dim))
    return da.copy(data=out.astype(da.dtype))


def _interp_at(x, where):
    """x (Series on a time index) with the positions `where` replaced by linear interpolation in time
    from the remaining values (nearest valid value at the ends)."""
    where = np.asarray(where, bool)
    if not where.any():
        return x
    y = x.mask(where)
    try:
        y = y.interpolate(method="time")
    except (TypeError, ValueError):
        y = y.interpolate()
    y = y.ffill().bfill()
    return x.where(~where, y)
MIN_PER_MONTH = 48               # valid values needed to judge a month

K0 = 273.15


def _f(factor):
    return lambda x: x * factor


def _off(offset):
    return lambda x: x + offset


# name: spec
#   unit      expected unit (stored in the attributes)
#   med       plausible range of the monthly statistic
#   stat      "median" (all values) | "median_pos" (values > 0: daytime radiation, rain events)
#   valid     physically possible range (outside -> NaN)
#   fixes     [(range of the statistic in the wrong unit, conversion, wrong unit)]
#   flat      constant runs are fill values (-> NaN)
#   era       FLUXNET counterpart used to fill masked values (same unit expected after checks)
#   monthly   False: judge the whole record only (no time axis, or month-to-month ambiguous)
_TA = dict(unit="degC", med=(-50.0, 45.0), valid=(-60.0, 60.0),
           fixes=[((180.0, 330.0), _off(-K0), "K")], flat=True)
# ERA5 temperatures: degC and K are both fine (the model converts to degC); the record keeps the
# unit most of it is in, only periods in the other unit are converted to it
# ranges cover hot topsoil (stl1 on bare soil at midday reaches 60+ degC) and cannot overlap
_TK = dict(unit="degC or K", either={"degC": (-70.0, 75.0), "K": (200.0, 350.0)},
           conv={("K", "degC"): _off(-K0), ("degC", "K"): _off(K0)},
           valid_by_unit={"degC": (-90.0, 70.0), "K": (183.0, 343.0)})
# monthly median of the daytime values: a few W m-2 in polar winter, ~1000 W m-2 at the equator
_SW = dict(unit="W m-2", stat="median_pos", med=(0.5, 1000.0), valid=(-30.0, 1500.0),
           fixes=[((0.0005, 1.0), _f(1000.0), "kW m-2")], flat=True)
_SWC = dict(unit="m3 m-3", med=(0.0, 0.8), valid=(0.0, 1.0), fixes=[((1.0, 80.0), _f(0.01), "%")])

VAR_SPECS = {
    # tower meteorology (FLUXNET names and the names in the site files)
    "t2m": dict(_TA, era="TA_ERA", era5_site="t2m_era5"),      # fill: TA_ERA if plausible, else ERA5-Land at the tower
    "TA_F": dict(_TA, era="TA_ERA"),
    "ssrd": dict(_SW, era="SW_IN_ERA"),
    "SW_IN_F": dict(_SW, era="SW_IN_ERA"),
    "SW_IN_POT": dict(_SW, med=(0.5, 1100.0), valid=(0.0, 1450.0), flat=False),
    "PA_F": dict(unit="kPa", med=(50.0, 110.0), valid=(40.0, 112.0), flat=True, era="PA_ERA",
                 fixes=[((500.0, 1100.0), _f(0.1), "hPa"), ((5.0e4, 1.1e5), _f(0.001), "Pa")]),
    # rain: judged on the mean annual sum (single events are no unit test: ERA-filled precipitation has
    # many tiny amounts); in mm 10 - 8000 mm per year at any site, in m 1000x less
    "P_F": dict(unit="mm per time step", stat="annual_sum", med=(10.0, 8000.0), valid=(0.0, 150.0), era="P_ERA",
                fixes=[((0.005, 8.0), _f(1000.0), "m")], monthly=False),
    "WS_F": dict(unit="m s-1", med=(0.05, 25.0), valid=(0.0, 60.0), flat=True, era="WS_ERA"),
    "USTAR": dict(unit="m s-1", med=(0.01, 2.5), valid=(0.0, 5.0), flat=True),
    "CO2_F_MDS": dict(unit="umol mol-1", med=(300.0, 900.0), valid=(200.0, 2000.0),
                      fixes=[((3e-4, 9e-4), _f(1e6), "mol mol-1")]),
    # energy fluxes
    "LE_F_MDS": dict(unit="W m-2", med=(-60.0, 350.0), valid=(-300.0, 1000.0)),
    "H_F_MDS": dict(unit="W m-2", med=(-250.0, 450.0), valid=(-400.0, 1200.0)),
    "LE_RANDUNC": dict(unit="W m-2", med=(0.0, 250.0), valid=(0.0, 600.0)),     # ~0 in frozen winters
    "LE_CORR": dict(unit="W m-2", med=(-60.0, 400.0), valid=(-300.0, 1200.0)),
    "NETRAD": dict(unit="W m-2", med=(-120.0, 650.0), valid=(-300.0, 1200.0)),
    "G_F_MDS": dict(unit="W m-2", med=(-120.0, 200.0), valid=(-300.0, 500.0)),
    # targets
    "NEE_VUT_REF": dict(unit="umol m-2 s-1", med=(-30.0, 20.0), valid=(-80.0, 60.0)),
    "NEE_VUT_REF_RANDUNC": dict(unit="umol m-2 s-1", med=(0.03, 20.0), valid=(0.0, 60.0)),
    # ERA5 (gridded; checked on the spatial mean)
    "t2m_era5": _TK,
    "stl1_era5": _TK,
    "stl2_era5": _TK,
    "swvl1_era5": _SWC,
    "swvl2_era5": _SWC,
    "sd_era5": dict(unit="m", med=(0.0, 10.0), valid=(0.0, 50.0)),
    # satellite indices (no calendar time axis here: whole record)
    "nirv": dict(unit="1", med=(-0.1, 0.8), valid=(-0.5, 1.5), monthly=False,
                 fixes=[((2.0, 8000.0), _f(1e-4), "scaled integer (x 1e4)")]),
    "lswi": dict(unit="1", med=(-1.0, 1.0), valid=(-1.01, 1.01), monthly=False,
                 fixes=[((1.5, 10000.0), _f(1e-4), "scaled integer (x 1e4)"),
                        ((-10000.0, -1.5), _f(1e-4), "scaled integer (x 1e4)")]),
}


# ---------------------------------------------------------------------------
# adapters: the same checks on an xarray Dataset (site file) or a pandas DataFrame (tower data)
# ---------------------------------------------------------------------------

class _DatasetAdapter:
    def __init__(self, ds):
        self.ds = ds
        self.times = pd.DatetimeIndex(ds[TIME_DIM].values) if TIME_DIM in ds.coords or TIME_DIM in ds else None
        self.lon = ds.attrs.get("site_lon")

    def has(self, v):
        return v in self.ds.variables

    def has_time(self, v):
        return TIME_DIM in self.ds[v].dims

    def series(self, v):
        """1-D series over time (spatial mean for gridded fields); whole-record values otherwise."""
        da = self.ds[v]
        if TIME_DIM in da.dims:
            other = [d for d in da.dims if d != TIME_DIM]
            da = da.mean(other, skipna=True) if other else da
            return pd.Series(np.asarray(da.values, dtype=np.float64), index=self.times)
        return pd.Series(np.asarray(da.values, dtype=np.float64).ravel())

    def site_series(self, v):
        """Series at the tower: gridded fields (e.g. ERA5-Land t2m_era5, vpd_era5) at the grid cell nearest to
        site_lat / site_lon that has valid data (> 50 % of the time steps); 1-D variables as they are."""
        cache = self.__dict__.setdefault("_site_cache", {})
        if v in cache:
            return cache[v]
        da = self.ds[v]
        if TIME_DIM not in da.dims or da.ndim == 1:
            cache[v] = self.series(v)
            return cache[v]
        lat0, lon0 = self.ds.attrs.get("site_lat"), self.ds.attrs.get("site_lon")
        names = [c for c in ("lat", "latitude") if c in da.coords], [c for c in ("lon", "longitude") if c in da.coords]
        if lat0 is None or lon0 is None or not names[0] or not names[1]:
            cache[v] = self.series(v)                 # no coordinates: spatial mean as before
            return cache[v]
        la, lo = da[names[0][0]], da[names[1][0]]
        sdims = [d for d in da.dims if d != TIME_DIM]
        valid = da.notnull().mean(TIME_DIM) > 0.5
        la2, lo2 = xr.broadcast(la, lo)
        dist = (la2 - float(lat0)) ** 2 + ((lo2 - float(lon0)) * np.cos(np.radians(float(lat0)))) ** 2
        dist = dist.transpose(*[d for d in sdims if d in dist.dims])
        valid = valid.transpose(*dist.dims)
        dv = np.where(valid.values, dist.values, np.inf)
        if not np.isfinite(dv).any():
            cache[v] = self.series(v)
            return cache[v]
        pos = np.unravel_index(int(np.argmin(dv)), dv.shape)
        cell = da.isel({d: int(i) for d, i in zip(dist.dims, pos)})
        cache[v] = pd.Series(np.asarray(cell.values, dtype=np.float64), index=self.times)
        return cache[v]

    def _keep_orig(self, v):
        if f"{v}_orig" not in self.ds.variables:
            self.ds[f"{v}_orig"] = self.ds[v].copy()

    def convert(self, v, func, time_mask=None):
        self._keep_orig(v)
        da = self.ds[v]
        if time_mask is None or not self.has_time(v):
            self.ds[v] = func(da).astype(da.dtype)
        else:
            m = self.ds[TIME_DIM].copy(data=np.asarray(time_mask))
            self.ds[v] = da.where(~m, func(da)).astype(da.dtype)

    def mask(self, v, time_mask):
        if not self.has_time(v):
            return
        self._keep_orig(v)
        m = self.ds[TIME_DIM].copy(data=np.asarray(time_mask))
        self.ds[v] = self.ds[v].where(~m)

    def fill(self, v, time_mask, values):
        if not self.has_time(v) or self.ds[v].ndim != 1:
            return
        self.ds[v] = self.ds[v].copy(data=np.where(time_mask, values, self.ds[v].values))

    def set_series(self, v, s):
        self._keep_orig(v)
        self.ds[v] = self.ds[v].copy(data=np.asarray(s, dtype=self.ds[v].dtype))

    def range_ops(self, v, lo, hi, e):
        """Element-wise on the full array: clip storage noise to the limits, then return
        (share of impossible values, min, median, max) and a function that masks them."""
        da = self.ds[v]
        da = da.where(da > -9990)
        tiny = ((da < lo) & (da >= lo - e)) | ((da > hi) & (da <= hi + e))
        n_tiny = float(tiny.mean())
        if n_tiny > 0:
            self._keep_orig(v)
            self.ds[v] = self.ds[v].where(~tiny, self.ds[v].clip(lo, hi))
            da = self.ds[v].where(self.ds[v] > -9990)
        bad = (da < lo - e) | (da > hi + e)
        valid = da.notnull()
        share = float(bad.sum() / max(int(valid.sum()), 1))
        self._last_counts = (int(bad.sum()), int(valid.sum()))
        vals = da.values[np.isfinite(da.values)]
        stats = (float(vals.min()), float(np.median(vals)), float(vals.max())) if vals.size else (np.nan,) * 3

        def apply_mask():
            self._keep_orig(v)
            if v in CLIP_VARS:
                self.ds[v] = self.ds[v].clip(lo, hi)
            elif v in NO_NAN_VARS:
                tdim = next((d for d in self.ds[v].dims if d in (TIME_DIM, "time_gap_filled", "time", "t")), None)
                x = self.ds[v].where(~bad)
                if tdim is not None and self.ds[v].sizes[tdim] > 1:
                    x = _interp_array_along(x, tdim)
                # anything still missing (no valid value of that pixel at all): the clipped original
                x = x.fillna(self.ds[v].clip(lo, hi)).clip(lo, hi)
                self.ds[v] = x.where(self.ds[v].notnull())          # pre-existing NaNs stay as they were
            else:
                self.ds[v] = self.ds[v].where(~bad)
        return n_tiny, share, stats, apply_mask

    def note(self, v, text):
        a = self.ds[v].attrs
        a["checks"] = (a.get("checks", "") + "; " + text).strip("; ")


class _FrameAdapter:
    def __init__(self, df, times=None, lon=None):
        self.df = df
        t = times if times is not None else (df["datetime_utc"] if "datetime_utc" in df else df.index)
        self.times = pd.DatetimeIndex(t)
        self.lon = lon

    def has(self, v):
        return v in self.df

    def has_time(self, v):
        return True

    def series(self, v):
        return pd.Series(pd.to_numeric(self.df[v], errors="coerce").values, index=self.times)

    def site_series(self, v):
        return self.series(v)

    def _keep_orig(self, v):
        if f"{v}_orig" not in self.df:
            self.df[f"{v}_orig"] = self.df[v].values

    def convert(self, v, func, time_mask=None):
        self._keep_orig(v)
        x = self.series(v)
        new = func(x)
        self.df[v] = (new if time_mask is None else x.where(~np.asarray(time_mask), new)).values

    def mask(self, v, time_mask):
        self._keep_orig(v)
        self.df[v] = self.series(v).where(~np.asarray(time_mask)).values

    def fill(self, v, time_mask, values):
        self.df[v] = np.where(time_mask, values, self.series(v).values)

    def set_series(self, v, s):
        self._keep_orig(v)
        self.df[v] = np.asarray(s)

    def range_ops(self, v, lo, hi, e):
        x = self.series(v).where(lambda z: z > -9990)
        tiny = (((x < lo) & (x >= lo - e)) | ((x > hi) & (x <= hi + e))).values
        if tiny.any():
            self.convert(v, lambda z: z.clip(lo, hi), tiny)
            x = self.series(v).where(lambda z: z > -9990)
        bad = ((x < lo - e) | (x > hi + e)).values
        share = float(bad.sum() / max(int(x.notna().sum()), 1))
        self._last_counts = (int(bad.sum()), int(x.notna().sum()))
        xs = x.dropna()
        stats = (float(xs.min()), float(xs.median()), float(xs.max())) if len(xs) else (np.nan,) * 3
        def apply_mask():
            if v in NO_NAN_VARS:
                self._keep_orig(v)
                self.df[v] = _interp_at(self.series(v), bad).clip(lo, hi).values
            else:
                self.mask(v, bad)
        return float(tiny.mean()), share, stats, apply_mask

    def note(self, v, text):
        pass


# ---------------------------------------------------------------------------
# single-variable checks
# ---------------------------------------------------------------------------

def _stat(x, kind):
    x = x[np.isfinite(x)]
    if kind == "median_pos":
        x = x[x > 0]
    return float(np.median(x)) if len(x) else np.nan


def _eps(rng):
    """Numerical tolerance at the range limits (ERA5 storage noise, e.g. snow depth -7e-25 m)."""
    return 1e-6 * max(abs(rng[1] - rng[0]), 1.0)


def _in(v, rng):
    e = _eps(rng)
    return np.isfinite(v) and rng[0] - e <= v <= rng[1] + e


# FLUXNET quality flags (0 = measured, > 0 = gap-filled, e.g. from ERA): units are judged separately
# for measured and gap-filled values, so a unit difference between the two is found within a day.
QC_VARS = {"t2m": "TA_F_QC", "TA_F": "TA_F_QC", "ssrd": "SW_IN_F_QC", "SW_IN_F": "SW_IN_F_QC",
           "PA_F": "PA_F_QC", "P_F": "P_F_QC", "WS_F": "WS_F_QC", "VPD_F": "VPD_F_QC",
           "VPD_F_MDS": "VPD_F_MDS_QC"}


def _qc_group(a, v):
    """0 = measured, 1 = gap-filled per time step; None if the variable has no QC flag."""
    q = QC_VARS.get(v)
    if not (q and a.has(q) and a.has_time(q)):
        return None
    qc = a.series(q).values
    return np.where(np.isfinite(qc) & (qc > 0), 1, 0)


def _find_fix(stat, spec):
    """(conversion, wrong unit) bringing stat into the plausible range; None if stat fits; 'bad' if nothing
    fits; ('ambiguous', unit) if stat fits as it is AND after a conversion (e.g. polar winter radiation)."""
    if not np.isfinite(stat):
        return None
    if _in(stat, spec["med"]):
        for rng, func, unit in spec.get("fixes", []):
            if _in(stat, rng) and _in(float(func(stat)), spec["med"]):
                return "ambiguous", unit
        return None
    for rng, func, unit in spec.get("fixes", []):
        if _in(stat, rng) and _in(float(func(stat)), spec["med"]):
            return func, unit
    return "bad"


def _check_units(a, v, spec, out):
    """Unit per DAY (variables judged on all values, >= 24 per day), then per MONTH for the days that
    could not be judged (too few values, or event statistics like rain / daytime radiation)."""
    x = a.series(v).where(lambda z: z > -9990)
    kind = spec.get("stat", "median")
    timed = spec.get("monthly", True) and a.has_time(v) and a.times is not None
    if not timed:
        ref_txt = ""
        if kind == "annual_sum":
            dt = (float(np.median(np.diff(a.times.values).astype("timedelta64[s]").astype(np.float64))) / 3600.0
                  if a.times is not None and len(a.times) > 1 else 0.5)                      # hours
            stat = float(np.nanmean(x.values)) * 8766.0 / dt            # mean annual sum (gaps excluded)
            era = spec.get("era")
            if era and a.has(era):
                e = a.series(era).where(lambda z: z > -9990)
                ok = x.notna() & e.notna()
                if ok.sum() > 1000:
                    e_sum = float(e[ok].mean()) * 8766.0 / dt
                    t_sum = float(x[ok].mean()) * 8766.0 / dt
                    ref_txt = f" ({era}: {e_sum:.0f} per year, tower / ERA = {t_sum / max(e_sum, 1e-9):.2f})"
                    if not 0.33 <= t_sum / max(e_sum, 1e-9) <= 3.0:
                        out.append(dict(level="warning", var=v, msg=f"annual sum {t_sum:.0f} vs {era} {e_sum:.0f}: "
                                        f"ratio {t_sum / max(e_sum, 1e-9):.2f} - double counting, aggregation or unit?"))
        else:
            stat = _stat(x.values, kind)
        res = _find_fix(stat, spec)
        if res is None or res[0] == "ambiguous":
            return
        if res == "bad":
            out.append(dict(level="warning" if kind == "annual_sum" else "error", var=v,
                            msg=f"{kind.replace('_', ' ')} {stat:.4g} outside the plausible range {spec['med']} and no "
                                f"known unit error fits{ref_txt}", value=stat))
            return
        a.convert(v, res[0])
        a.note(v, f"record in {res[1]}, converted to {spec['unit']}")
        out.append(dict(level="correction", var=v, msg=f"record in {res[1]}, converted to {spec['unit']}"))
        return

    vals = x.values
    qg = _qc_group(a, v)
    g01 = np.zeros(len(vals), np.int64) if qg is None else qg.astype(np.int64)
    # integer period codes (x 2 + QC group: measured and gap-filled judged separately)
    day_code = a.times.values.astype("datetime64[D]").astype(np.int64) * 2 + g01          # days since 1970
    mon_code = (a.times.year.values.astype(np.int64) * 12 + a.times.month.values - 1) * 2 + g01
    month_of = a.times.to_period("M")
    judged = np.zeros(len(vals), bool)
    ambiguous = np.zeros(len(vals), bool)
    implausible = np.zeros(len(vals), bool)                   # periods fitting no unit at all
    unit_of = np.full(len(vals), None, dtype=object)          # wrong unit per time step
    func_of = {}
    bad_periods = []
    n_periods = 0
    stat_vals = np.where(vals > 0, vals, np.nan) if kind == "median_pos" else vals

    def judge(codes, sel, min_n, label):
        """Judge all periods (codes) among the time steps `sel` at once; returns the codes judged
        as 'not fine' (wrong unit, ambiguous or implausible)."""
        nonlocal n_periods
        s_ = pd.Series(stat_vals[sel])
        grp = s_.groupby(codes[sel])
        med, cnt = grp.median(), grp.count()
        ok = cnt.index[cnt >= min_n]
        n_periods += len(ok)
        flagged, bad_c, amb_c, unit_c = [], [], [], {}
        for c in ok:
            res = _find_fix(float(med[c]), spec)
            if res is None:
                continue
            flagged.append(c)
            if res == "bad":
                bad_periods.append((label(c), float(med[c])))
                bad_c.append(c)
            elif res[0] == "ambiguous":
                amb_c.append(c)
            else:
                unit_c.setdefault(res[1], []).append(c)
                func_of[res[1]] = res[0]
        # a conversion must give physically possible values: if more than 1 % of a period would lie outside
        # the possible range afterwards (e.g. a broken sensor reading ~0 with a few real values, taken for
        # kW m-2), it is not a unit error - the period is only flagged as implausible
        lo_v, hi_v = spec["valid"]
        for u in list(unit_c):
            cs = np.asarray(unit_c[u])
            m = sel & np.isin(codes, cs)
            conv = np.asarray(func_of[u](vals[m]), dtype=np.float64)
            outside = pd.Series((conv < lo_v) | (conv > hi_v)).groupby(codes[m]).mean()
            rejected = outside.index[outside > 0.01]
            if len(rejected):
                for c in rejected:
                    bad_periods.append((label(c) + f" (would be impossible in {spec['unit']} after the {u} conversion)",
                                        float(med[c])))
                bad_c.extend(rejected.tolist())
                unit_c[u] = [c for c in cs if c not in set(rejected)]
                if not unit_c[u]:
                    del unit_c[u]
        # apply the verdicts once per category (vectorised)
        if bad_c:
            m = sel & np.isin(codes, bad_c)
            ambiguous[m], unit_of[m] = False, None
            implausible[m] = True
        if amb_c:
            m = sel & np.isin(codes, amb_c)
            ambiguous[m], unit_of[m] = True, None
        for u, cs in unit_c.items():
            m = sel & np.isin(codes, cs)
            ambiguous[m], unit_of[m] = False, u
        judged[sel & np.isin(codes, ok)] = True
        return flagged

    mon_label = lambda c: f"{(c // 2) // 12}-{(c // 2) % 12 + 1:02d}" + ("" if qg is None else ("|filled" if c % 2 else "|measured"))
    day_label = lambda c: str(np.datetime64(int(c // 2), "D")) + ("" if qg is None else ("|filled" if c % 2 else "|measured"))

    # coarse to fine: months first; days only inside months that look wrong or hold impossible values
    everything = np.ones(len(vals), bool)
    flagged_months = judge(mon_code, everything, 20 if kind == "median_pos" else MIN_PER_MONTH, mon_label)
    if kind == "median":
        lo, hi = spec["valid"]
        impossible = (vals < lo) | (vals > hi)
        susp = set(flagged_months) | set(np.unique(mon_code[impossible]).tolist())
        if susp:
            in_susp = np.isin(mon_code, list(susp))
            # reset the month verdict there, the days decide (too short days keep it below)
            month_unit, month_amb, month_judged = unit_of.copy(), ambiguous.copy(), judged.copy()
            month_impl = implausible.copy()
            bad_periods[:] = [b for b in bad_periods if not any(b[0].split(" (")[0] == mon_label(c) for c in susp)]
            unit_of[in_susp], ambiguous[in_susp], judged[in_susp] = None, False, False
            implausible[in_susp] = False
            judge(day_code, in_susp, 12 if qg is not None else 24, day_label)
            # days too short to judge: back to the verdict of their month
            short = in_susp & ~judged & month_judged
            unit_of[short], ambiguous[short], judged[short] = month_unit[short], month_amb[short], True
            implausible[short] = month_impl[short]
    valid = np.isfinite(vals)
    # periods too short to judge take the unit of their group (measured / gap-filled, or the whole
    # record without QC flags) if at least 90 % of the judged values of that group agree
    groups = [np.ones(len(vals), bool)] if qg is None else [qg == 0, qg == 1]
    for grp in groups:
        unj = grp & ~judged & valid
        jud = grp & judged & valid & ~ambiguous
        if unj.any() and jud.sum() >= MIN_PER_MONTH:
            lab = np.array([u if u is not None else "" for u in unit_of[jud]])
            units, counts = np.unique(lab, return_counts=True)
            top = units[np.argmax(counts)]
            if counts.max() / counts.sum() >= 0.9 and top != "":
                unit_of[unj] = top
                judged |= unj
    # ambiguous periods take the unit of the majority of the unambiguous ones
    if ambiguous.any():
        clear = judged & ~ambiguous & valid
        units, counts = np.unique([u if u is not None else "" for u in unit_of[clear]], return_counts=True)
        if len(units) and units[np.argmax(counts)] != "":
            unit_of[ambiguous] = units[np.argmax(counts)]
    for unit, func in func_of.items():
        m = unit_of == unit
        if not m.any():                       # all periods of this unit rejected above
            continue
        share = float(m[valid].mean()) if valid.any() else 0.0
        # every judged period in this unit -> the whole record (incl. periods too short to judge)
        jv = judged & valid
        whole = (float(m[jv].mean()) if jv.any() else 0.0) > 0.99
        a.convert(v, func, None if whole else m)
        months = sorted({str(p) for p in month_of[m]})
        which = ""
        if qg is not None and not whole:
            f_filled = float((qg[m] == 1).mean())
            which = (" - only gap-filled values" if f_filled > 0.99 else
                     " - only measured values" if f_filled < 0.01 else f" - {f_filled:.0%} of them gap-filled")
        msg = (f"record in {unit}, converted to {spec['unit']}" if whole else
               f"UNIT CHANGE WITHIN THE RECORD: {share:.0%} of the record in {unit} "
               f"({months[0]} .. {months[-1]}){which}, converted to {spec['unit']}")
        a.note(v, msg)
        out.append(dict(level="correction" if whole else "warning", var=v, msg=msg, share=share,
                        months=months[:6] + (["..."] if len(months) > 6 else [])))
    if bad_periods:
        share = len(bad_periods) / max(n_periods, 1)
        # implausible periods (no unit fits) are replaced by the ERA counterpart where there is one
        era, e = _era_ref(a, v, spec, out)
        fill_txt = ""
        if e is not None and implausible.any() and a.series(v).ndim == 1:
            f = implausible & e.notna().values
            if f.any():
                a.set_series(v, a.series(v).where(~f, e).values)
                fill_txt = f"; these {f.mean():.2%} of the record replaced by {era}"
                a.note(v, f"{f.mean():.2%} of the record in implausible periods replaced by {era}")
        out.append(dict(level="error" if (share > 0.3 and not fill_txt) else "warning", var=v,
                        msg=f"{len(bad_periods)} periods (day / month) outside the plausible range for "
                            f"{spec['unit']} {spec['med']}, no known unit error fits: "
                            + ", ".join(f"{k} {val:.4g}" for k, val in bad_periods[:4])
                            + (" ..." if len(bad_periods) > 4 else "") + fill_txt,
                        share=share, examples=bad_periods[:4], changed=bool(fill_txt)))


def _era_ref(a, v, spec, out):
    """The ERA counterpart of v if it is plausible in v's unit (statistic within the plausible range,
    < 1 % outside the possible range), else None (with one warning). Used before any ERA fill."""
    if not FILL_FROM_ERA:
        return None, None
    cache = a.__dict__.setdefault("_era_cache", {})
    # candidates in order: the FLUXNET *_ERA variable (downscaled and bias-corrected against the site's own
    # measurements, the source of FLUXNET's gap filling), then ERA5-Land at the tower pixel (raw 9 km product)
    for era, site in ((spec.get("era"), False), (spec.get("era5_site"), True)):
        if not (era and a.has(era)):
            continue
        if era not in cache:
            e = (a.site_series(era) if site else a.series(era)).where(lambda z: z > -9990)
            if site and spec["unit"] == "degC" and np.nanmedian(e) > 150:
                e = e - K0                                         # ERA5 temperature in K
            kind = spec.get("stat", "median")
            if kind == "annual_sum":
                cache[era] = e
            else:
                st = _stat(e.values, kind)
                lo, hi = spec["valid"]
                out_share = float(((e < lo) | (e > hi)).mean())
                if not _in(st, spec["med"]) or out_share > 0.01:
                    out.append(dict(level="warning", var=era, msg=f"{era} not plausible in {spec['unit']} "
                                    f"({kind.replace('_', ' ')} {st:.4g}, {out_share:.1%} outside {spec['valid']}) - "
                                    f"not used to fill {v}"))
                    cache[era] = None
                else:
                    cache[era] = e
        if cache[era] is not None:
            return era + (" (ERA5-Land at the tower)" if site else ""), cache[era]
    return None, None


def _check_either_units(a, v, spec, out):
    """Variables valid in two units (ERA5 temperature): find the unit of each day (month as fallback),
    keep the unit of the majority, convert periods in the other unit. Returns the spec for the
    range check in the record's unit."""
    x = a.series(v).where(lambda z: z > -9990)
    vals = x.values
    month = a.times.to_period("M")
    # the two units do not overlap (degC -70..75, K 200..350): every value is judged on its own
    unit_of = np.full(len(vals), None, dtype=object)
    for u, rng in spec["either"].items():
        unit_of[(vals >= rng[0]) & (vals <= rng[1])] = u
    n_neither = int((np.isfinite(vals) & (unit_of == None)).sum())       # noqa: E711
    bad = [("values", n_neither)] if n_neither else []
    units, counts = np.unique([u for u in unit_of if u is not None], return_counts=True)
    if not len(units):
        out.append(dict(level="error", var=v, msg=f"neither {list(spec['either'])}: median "
                        f"{float(np.nanmedian(vals)):.4g}, min {float(np.nanmin(vals)):.4g}, max {float(np.nanmax(vals)):.4g}"))
        return None
    main = units[np.argmax(counts)]
    for u in units:
        if u == main:
            continue
        m = unit_of == u
        share = float(m[np.isfinite(vals)].mean())
        a.convert(v, spec["conv"][(u, main)], m)
        months = sorted({str(p) for p in month[m]})
        msg = (f"UNIT CHANGE WITHIN THE RECORD: {share:.0%} of the record in {u}, converted to the record's "
               f"unit {main} ({months[0]} .. {months[-1]})")
        a.note(v, msg)
        out.append(dict(level="warning", var=v, msg=msg, share=share))
    if bad:
        out.append(dict(level="warning", var=v, msg=f"{n_neither} values fit neither {list(spec['either'])} "
                        "(left to the range check)"))
    a.note(v, f"unit {main}")
    return dict(spec, unit=main, valid=spec["valid_by_unit"][main])


def _check_valid(a, v, spec, out):
    """Values outside the physically possible range -> NaN, element by element (also gridded ERA5
    and satellite fields); storage noise just outside a limit -> the limit. More impossible values
    than MAX_IMPOSSIBLE_SHARE: error, nothing masked."""
    lo, hi = spec["valid"]
    e = _eps(spec["valid"])
    if v in CLIP_VARS and isinstance(a, _DatasetAdapter):
        # spurious values only in single pixels / dates are fine to clip; the 1st / 99th percentiles
        # show whether the bulk of the field is plausible
        vals = np.asarray(a.ds[v].values, dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > -9990)]
        if vals.size:
            p1, p99 = (float(q) for q in np.percentile(vals, [1, 99]))
            plo, phi = spec["med"]
            ok = plo <= p1 and p99 <= phi
            a._pct_note = f"; 1st / 99th percentile {p1:.3f} / {p99:.3f} (bulk plausible)" if ok else ""
            out.append(dict(level="info" if ok else "warning", var=v,
                            msg=f"1st / 99th percentile {p1:.3f} / {p99:.3f}"
                                + ("" if ok else f" - outside the plausible range {spec['med']}: widespread problem, "
                                                 "not single spurious values"), p1=p1, p99=p99))
    n_tiny, share, (mn, md, mx), apply_mask = a.range_ops(v, lo, hi, e)
    if n_tiny > 0:
        out.append(dict(level="info", var=v, msg=f"{n_tiny:.1%} values within {e:.0e} of the range limits "
                        f"set to the limit (numerical noise)", changed=True))
    if share == 0:
        return
    rng = f"{spec['valid']} {spec['unit']} (min {mn:.4g}, median {md:.4g}, max {mx:.4g})"
    if share > MAX_IMPOSSIBLE_SHARE and v not in CLIP_VARS:
        out.append(dict(level="error", var=v, msg=f"{share:.1%} of values outside the possible range {rng} "
                        "- not masked, check the source", share=share))
        return
    apply_mask()
    how = ("clipped to the range" if v in CLIP_VARS else
           "replaced by interpolation in time (network input, never NaN)" if v in NO_NAN_VARS else "set to NaN")
    a.note(v, f"{share:.3%} values outside {spec['valid']} {how}")
    nb, nt = getattr(a, "_last_counts", (0, 0))
    how = how + (getattr(a, "_pct_note", "") if v in CLIP_VARS else "")
    out.append(dict(level="correction", var=v, msg=f"{nb} of {nt} values ({share:.2g}) outside the possible range {rng} "
                    + how, share=share))


def _flat_runs(x, steps=FLAT_STEPS):
    run_id = (x != x.shift()).cumsum()
    run_len = x.groupby(run_id).transform("size")
    return ((run_len >= steps) & x.notna() & (x != 0)).values


def _check_flat(a, v, spec, out):
    if not (spec.get("flat") and a.has_time(v)):
        return
    x = a.series(v)
    if v in ("ssrd", "SW_IN_F", "SW_IN_POT"):
        x = x.where(x > 5)                               # night zeros are constant by nature
    flat = _flat_runs(x)
    share = float(flat.mean())
    if share < 0.001:
        return
    a.mask(v, flat)
    filled = 0.0
    era, e = _era_ref(a, v, spec, out)
    if e is not None:
        f = flat & e.notna().values
        a.fill(v, f, e.values)
        filled = float(f.mean())
    left = flat & ~np.isfinite(a.series(v).values)
    interp = 0.0
    if v in TOWER_INPUTS and left.any():
        a.set_series(v, _interp_at(a.series(v), left).values)
        interp, left = float(left.mean()), np.zeros_like(left)
    parts = []
    if filled:
        parts.append(f"{filled:.2%} replaced by {era}")
    if interp:
        parts.append(f"{interp:.2%} interpolated in time (network input, never NaN)")
    if left.any():
        parts.append(f"{left.mean():.2%} set to NaN" + ("" if e is not None else " (no usable ERA counterpart)"))
    msg = f"{share:.2%} of the record in constant runs (>= {FLAT_STEPS} identical values): " + ", ".join(parts)
    a.note(v, msg)
    out.append(dict(level="correction", var=v, msg=msg, share=share, filled=filled))


# ---------------------------------------------------------------------------
# VPD against the saturation vapour pressure
# ---------------------------------------------------------------------------

def magnus_hpa(ta_c):
    return 6.1078 * np.exp(17.27 * ta_c / (ta_c + 237.3))


def clean_vpd(vpd, ta_c, times, vpd_era=None, flat_steps=FLAT_STEPS, pa_ratio=3.0, max_ratio=1.1, qc_group=None,
              vpd_ref=None):
    """
    VPD in hPa, robust to unit changes WITHIN a record (e.g. CD-Ygb: Pa first, hPa later).
    1. Record median < 0.6 -> kPa (x 10). Then per month and per day: median VPD / e_s(TA) > pa_ratio
       is impossible in hPa -> Pa (x 0.01).
    2. VPD > max_ratio * e_s after the conversion -> NaN.
    3. Constant runs (>= flat_steps identical non-zero values) -> NaN.
    4. Gaps from 2.-3. filled from vpd_era where given.
    qc_group: optional 0 (measured) / 1 (gap-filled) per time step; the Pa test is then done separately
    for measured and gap-filled values of each day / month.
    Returns (vpd_hpa Series, report dict).
    """
    idx = pd.DatetimeIndex(times)
    x = pd.Series(np.asarray(vpd, dtype=np.float64), index=idx).where(lambda z: z > -9990)
    es = magnus_hpa(pd.Series(np.asarray(ta_c, dtype=np.float64), index=idx))
    rep = {"months_converted_from_Pa": [], "days_converted_from_Pa": 0, "record_converted_from_kPa": False}
    # kPa only if VPD is small in absolute terms AND relative to the air's capacity: cold, humid sites
    # (e.g. CA-CF3, subarctic fen) have median VPD < 0.6 hPa in hPa, but VPD / e_s ~ 0.1-0.3, while kPa
    # gives ~10x less. With a reference (VPD_ERA, hPa) the record median ratio decides instead.
    med_abs = float(x.where(x > 0).median())
    if med_abs < 0.6:
        ref = None
        if vpd_ref is not None:
            e = pd.Series(np.asarray(vpd_ref, dtype=np.float64), index=idx).where(lambda z: z > -9990)
            if float(e.where(e > 0).median()) > 60:
                e = e * 0.01
            ok = (x > 0) & (e > 0)
            if ok.sum() > 1000:
                ref = float(x[ok].median() / e[ok].median())
        rel = float((x / es).where(x > 0).median())
        rep["kpa_test"] = dict(median=med_abs, median_vpd_over_es=rel, median_ratio_to_era=ref)
        is_kpa = (0.03 < ref < 0.3) if ref is not None else rel < 0.04
        if is_kpa:
            x = x * 10.0
            rep["record_converted_from_kPa"] = True
    ratio = (x / es).where(x > 0)
    month = idx.to_period("M")
    day_k = pd.Index(idx.normalize().astype(str))
    mon_k = pd.Index(month.astype(str))
    if qc_group is not None:
        tag = np.where(np.asarray(qc_group) == 1, "|filled", "|measured")
        day_k, mon_k = day_k + tag, mon_k + tag
    dn = ratio.groupby(day_k).count()
    dmed = ratio.groupby(day_k).median()
    judged_days = dn.index[dn >= (6 if qc_group is not None else 12)]
    pa_days = dmed.index[(dmed > pa_ratio) & dn.index.isin(judged_days)]
    in_pa = day_k.isin(pa_days)
    # days with too few values: judged by their month (computed on those days' ratio)
    rest = ~day_k.isin(judged_days)
    if rest.any():
        mmed = ratio[rest].groupby(mon_k[rest]).median()
        pa_m = mmed.index[mmed > pa_ratio]
        in_pa |= rest & mon_k.isin(pa_m)
    in_pa = np.asarray(in_pa)
    day = idx.normalize()
    rep["days_converted_from_Pa"] = int(len(pd.Index(day[in_pa]).unique()))
    rep["months_converted_from_Pa"] = sorted({str(m) for m in month[in_pa]})
    rep["share_converted_from_Pa"] = float(in_pa[x.notna().values].mean()) if x.notna().any() else 0.0
    x = x.where(~in_pa, x * 0.01)
    # VPD cannot be negative: noise just below 0 (sensor at saturation, ERA5) -> 0; clearly negative -> impossible
    tiny_neg = (x < 0) & (x >= -1.0)
    rep["negative_set_to_zero_share"] = float(tiny_neg.mean())
    x = x.where(~tiny_neg, 0.0)
    impossible = ((x > max_ratio * es) & es.notna()) | (x < -1.0)
    rep["impossible_share"] = float(impossible.mean())
    x = x.mask(impossible)
    flat = _flat_runs(x, flat_steps)
    rep["flat_share"] = float(flat.mean())
    x = x.mask(flat)
    rep["filled_from_era_share"] = 0.0
    rep["gaps_filled_share"] = 0.0
    was_gap = ~np.isfinite(np.asarray(vpd, dtype=np.float64)) | (np.asarray(vpd, dtype=np.float64) <= -9990)
    if vpd_era is not None:
        e = pd.Series(np.asarray(vpd_era, dtype=np.float64), index=idx).where(lambda z: z > -9990)
        if float(e.where(e > 0).median()) > 60:
            e = e * 0.01
        fill = x.isna() & e.notna()
        rep["filled_from_era_share"] = float((fill & ~was_gap).mean())       # values masked above
        rep["gaps_filled_share"] = float((fill & was_gap).mean())            # gaps already in the file
        x = x.where(~fill, e)
    rep["missing_share"] = float(x.isna().mean())
    return x, rep


def _vpd_ref(a, ta_c, tower=None):
    """Reference / fill series for VPD in hPa: the FLUXNET VPD_ERA first (site-corrected), then vpd_era5
    (ERA5-Land at the tower pixel) as fallback. A candidate is used only if, in one of hPa / Pa / kPa, it is physically a VPD:
    it rarely exceeds the saturation vapour pressure of its own temperature (< 15 %), median 0.05 - 60 hPa and a
    plausible median VPD / e_s. (No absolute night-time criterion: at dry sites VPD stays high at night.)
    (CD-Ygb: VPD_ERA in Pa with a hard floor at 12.7 hPa - no VPD at all.)
    tower: measured tower VPD in hPa (QC 0, units already checked); if given, a candidate must resemble it:
    median within x3, and not much drier in humid conditions (10th percentile of VPD / e_s at most 0.25 above
    the tower's). Without measured values: RH >= 70 % in >= 1 % of the time steps.
    Returns (Series in hPa, label, [reasons for rejected candidates])."""
    rejected = []
    es_tower = magnus_hpa(ta_c)
    t_ok = tower is not None and int(tower.notna().sum()) >= 1000
    if t_ok:
        rel_t10 = float((tower / es_tower).quantile(0.10))
        med_t = float(tower.median())
    # each reference is tested against ITS OWN temperature (VPD_ERA vs TA_ERA, vpd_era5 vs t2m_era5 at the
    # tower): a reference VPD belongs to the reference temperature, not to the tower's
    own_ta = {"vpd_era5": ("t2m_era5", True), "VPD_ERA": ("TA_ERA", False)}
    # FLUXNET VPD_ERA first (site-corrected, consistent with FLUXNET's own gap filling), ERA5-Land as fallback
    for name, site in (("VPD_ERA", False), ("vpd_era5", True)):
        if not a.has(name):
            continue
        es = es_tower
        tname, tsite = own_ta[name]
        if a.has(tname):
            t_own = (a.site_series(tname) if tsite else a.series(tname)).where(lambda z: z > -9990)
            t_own = t_own - K0 if np.nanmedian(t_own) > 150 else t_own
            if t_own.notna().mean() > 0.5:
                es = magnus_hpa(t_own)
        raw = (a.site_series(name) if site else a.series(name)).where(lambda z: z > -9990)
        if raw.notna().sum() < 1000:
            rejected.append(f"{name} (almost) empty")
            continue
        best = None
        passing = []
        for factor, unit in ((1.0, "hPa"), (0.01, "Pa"), (10.0, "kPa")):
            e = raw * factor
            med, p5 = float(e.median()), float(e.quantile(0.05))
            over = float(((e > 1.1 * es + 0.5) & e.notna() & es.notna()).mean())
            # relative to the air's capacity: median VPD / e_s is ~0.1 (cold, humid) to ~0.7 (desert) in the
            # right unit, 10x less in kPa read as hPa, 100x more in Pa read as hPa
            rel = float((e / es).where(e > 0).median())
            # humid moments must exist: a true VPD gets small relative to e_s at least sometimes (RH >= 70 % in
            # >= 1 % of the time steps - nights, rain, fog; also at dry sites). CD-Ygb's VPD_ERA never does
            # (floor at 12.7 hPa, implied RH never above ~50 % at a tropical forest site)
            if t_ok:
                both = e.notna() & tower.notna()
                ratio_med = float(e[both].median() / max(tower[both].median(), 0.05)) if both.sum() > 500 else med / max(med_t, 0.05)
                humid_ok = float((e / es).quantile(0.10)) - rel_t10 <= 0.25 and 1 / 3 <= ratio_med <= 3
            else:
                humid_ok = float((e / es).quantile(0.01)) < 0.3
            # up to 15 % above e_s allowed: at hot, dry sites small temperature offsets between the reference
            # and the tower (or ERA) give exceedances on afternoons
            if 0.05 <= med <= 60 and over < 0.15 and 0.01 <= rel <= 0.95 and humid_ok:
                passing.append((abs(np.log(rel / 0.3)), e, unit))
        if passing:                                   # the unit closest to a typical VPD / e_s wins
            _, e_best, u_best = min(passing, key=lambda z: z[0])
            best = (e_best, u_best)
        label = name + (" (ERA5-Land at the tower)" if site else "")
        if best is None:
            e = raw * (0.01 if float(raw.median()) > 60 else 1.0)
            tl = tname if a.has(tname) else "tower TA"
            cmp = (f", tower median {med_t:.1f} hPa, 10th pct of VPD/e_s {float((e / es).quantile(0.10)):.2f} vs tower "
                   f"{rel_t10:.2f}" if t_ok else f", implied RH never above {100 * (1 - float((e / es).quantile(0.01))):.0f} %")
            rejected.append(f"{label} is not a plausible VPD in hPa, Pa or kPa (as hPa: median {float(e.median()):.1f} hPa, "
                            f"{float(((e > 1.1 * es + 0.5) & e.notna()).mean()):.0%} above e_s({tl}){cmp}) - not used")
            continue
        rejected = [r_ + (" (note: FLUXNET used it to fill VPD_F)" if r_.startswith("VPD_ERA") else "")
                    for r_ in rejected]
        return best[0], f"{label}, stored in {best[1]}", rejected
    return None, None, rejected


def _check_vpd(a, out, ta_name):
    for v in ("VPD_F", "VPD_F_MDS"):
        if not a.has(v) or not a.has(ta_name):
            continue
        ta = a.series(ta_name)
        ta = ta - K0 if np.nanmedian(ta) > 150 else ta
        raw = a.series(f"{v}_orig") if a.has(f"{v}_orig") else a.series(v)
        qg_pre = _qc_group(a, v)
        tower_pre, _ = clean_vpd(raw.values, ta.values, a.times, vpd_era=None, qc_group=qg_pre, vpd_ref=None)
        measured = tower_pre.where((qg_pre == 0) if qg_pre is not None else np.ones(len(tower_pre), bool))
        e_ok, ref_label, why = _vpd_ref(a, ta, tower=measured)
        if not getattr(a, "_vpd_ref_reported", False):
            for w in why:
                out.append(dict(level="warning", var="VPD reference", msg=w))
            out.append(dict(level="info", var="VPD reference", msg=f"reference / fill for VPD: {ref_label or 'none usable'}"))
            a._vpd_ref_reported = True
        era = e_ok.values if (FILL_FROM_ERA and e_ok is not None) else None
        ref = e_ok.values if e_ok is not None else None
        # FLUXNET filled the gaps of VPD_F from VPD_ERA. If VPD_ERA is no plausible VPD (CD-Ygb) but ERA5-Land
        # at the tower is, all gap-filled values (VPD_*_QC > 0) are replaced by ERA5-Land
        qg = _qc_group(a, v)
        fluxnet_era_bad = a.has("VPD_ERA") and any(w.startswith("VPD_ERA") for w in why)
        refill_share = 0.0
        if FILL_FROM_ERA and fluxnet_era_bad and e_ok is not None and ref_label.startswith("vpd_era5"):
            if qg is not None:
                gf = (qg == 1) & e_ok.notna().values
                refill_share = float(gf.mean())
                raw = raw.where(~gf, e_ok)
            elif not getattr(a, "_vpd_refill_warned", False):
                out.append(dict(level="warning", var=v, msg="VPD_ERA is not usable, but without a QC flag the values "
                                "FLUXNET gap-filled from it cannot be identified - they stay in the record"))
                a._vpd_refill_warned = True
        new, r = clean_vpd(raw.values, ta.values, a.times, vpd_era=era, qc_group=qg, vpd_ref=ref)
        r["refilled_share"] = refill_share
        changed = (r["months_converted_from_Pa"] or r["days_converted_from_Pa"] or r["record_converted_from_kPa"]
                   or r["impossible_share"] > 0 or r["flat_share"] > 0 or r.get("gaps_filled_share", 0) > 0
                   or refill_share > 0 or r.get("negative_set_to_zero_share", 0) > 0)
        if v in TOWER_INPUTS:
            newly = np.isfinite(raw.values) & ~np.isfinite(new.values)
            r["interpolated_share"] = float(newly.mean())
            new = _interp_at(new, newly)
        if not changed or np.allclose(a.series(v).values, new.values, equal_nan=True):
            continue
        a.set_series(v, new.values)
        parts = []
        if r.get("negative_set_to_zero_share"):
            parts.append(f"{r['negative_set_to_zero_share']:.2%} slightly negative values (> -1 hPa) set to 0")
        if r.get("refilled_share"):
            parts.append(f"all gap-filled values ({r['refilled_share']:.1%} of the record, filled by FLUXNET from the "
                         f"unusable VPD_ERA) replaced by {ref_label}")
        if r["record_converted_from_kPa"]:
            parts.append("record in kPa")
        if r["months_converted_from_Pa"]:
            m = r["months_converted_from_Pa"]
            parts.append(f"{r['share_converted_from_Pa']:.0%} of the record in Pa "
                         f"({r['days_converted_from_Pa']} days, {m[0]} .. {m[-1]})")
        if r["impossible_share"] > 0:
            parts.append(f"{r['impossible_share']:.2%} > e_s(TA) masked")
        if r["flat_share"] > 0:
            parts.append(f"{r['flat_share']:.2%} constant runs masked")
        if r["filled_from_era_share"]:
            parts.append(f"these {r['filled_from_era_share']:.2%} replaced by {ref_label}")
        if r.get("gaps_filled_share"):
            parts.append(f"{r['gaps_filled_share']:.1%} gaps already in the file filled from {ref_label}")
        if r.get("interpolated_share"):
            parts.append(f"{r['interpolated_share']:.1%} interpolated in time")
        msg = "checked against e_s(TA): " + "; ".join(parts)
        a.note(v, msg)
        within = 0.0 < r["share_converted_from_Pa"] < 0.99
        out.append(dict(level="warning" if within else "correction", var=v,
                        msg=("UNIT CHANGE WITHIN THE RECORD - " if within else "") + msg, **r))


# ---------------------------------------------------------------------------
# cross-checks
# ---------------------------------------------------------------------------

def _diurnal_lag(sw, pot, times):
    """Lag [h] of measured against potential radiation from their clear-sky envelopes (95th percentile per
    time of day): circular cross-correlation over the whole daytime curve, refined between the steps.
    Clouds that prefer mornings or afternoons do not shift the envelope, unlike the mean or its peak."""
    h = np.round((times.hour + times.minute / 60.0) * 2) / 2
    env_s = pd.Series(sw.values, index=h).groupby(level=0).quantile(0.95)
    env_p = pd.Series(pot.values, index=h).groupby(level=0).quantile(0.95)
    idx = env_p.index.union(env_s.index)
    s_, p_ = env_s.reindex(idx).fillna(0).values, env_p.reindex(idx).fillna(0).values
    if s_.max() <= 0 or p_.max() <= 0 or len(idx) < 12:
        return np.nan
    step = float(np.median(np.diff(idx.values)))
    n = len(idx)
    lags = np.arange(-n // 4, n // 4 + 1)
    cc = np.array([np.corrcoef(np.roll(p_, k), s_)[0, 1] for k in lags])
    i = int(np.nanargmax(cc))
    off = 0.0
    if 0 < i < len(cc) - 1:                                        # parabolic refinement
        den = cc[i - 1] - 2 * cc[i] + cc[i + 1]
        off = 0.5 * (cc[i - 1] - cc[i + 1]) / den if den != 0 else 0.0
    return float((lags[i] + off) * step)


def _peak_hour(x, times):
    """Hour of day (UTC, fractional) of the maximum of the mean diurnal cycle."""
    h = times.hour + times.minute / 60.0
    prof = pd.Series(x.values, index=h).groupby(level=0).mean()
    return float(prof.idxmax()) if prof.notna().any() else np.nan


def _hdiff(a, b):
    return (a - b + 12.0) % 24.0 - 12.0


def _cross_checks(a, out, ta_name, sw_name):
    t = a.times
    lon = a.lon
    if t is None:
        return
    # time axis
    if t.has_duplicates:
        out.append(dict(level="error", var=TIME_DIM, msg="duplicate time stamps", n=int(t.duplicated().sum())))
    if not t.is_monotonic_increasing:
        out.append(dict(level="error", var=TIME_DIM, msg="time stamps not sorted"))
    steps = pd.Series(t).diff().dropna()
    if len(steps) and steps.mode().iloc[0] != steps.median():
        out.append(dict(level="warning", var=TIME_DIM, msg="irregular time step", mode=str(steps.mode().iloc[0])))

    # UTC: solar noon from potential and measured radiation vs longitude
    if a.has("SW_IN_POT"):
        pot = a.series("SW_IN_POT")
        p_peak = _peak_hour(pot, t)
        if lon is not None and np.isfinite(p_peak):
            d = _hdiff(p_peak, (12.0 - float(lon) / 15.0) % 24.0)
            if abs(d) > 0.75:
                out.append(dict(level="error", var="SW_IN_POT", msg=f"solar noon {d:+.1f} h off the site longitude "
                                "- time stamps not in UTC?", peak_utc=p_peak, lon=lon))
        if a.has(sw_name):
            sw = a.series(sw_name)
            d = _diurnal_lag(sw, pot, t)
            if np.isfinite(d) and abs(d) >= 0.75:
                out.append(dict(level="error", var=sw_name, msg=f"clear-sky envelope of the measured radiation "
                                f"lags the potential radiation by {d:+.2f} h - tower data shifted in time?", shift_h=d))
            elif np.isfinite(d) and abs(d) >= 0.4:
                out.append(dict(level="warning", var=sw_name, msg=f"clear-sky envelope of the measured radiation "
                                f"lags the potential radiation by {d:+.2f} h", shift_h=d))
            night = pot <= 0
            if night.sum() > 100:
                n_med = float(np.nanmedian(sw[night]))
                if n_med > 10:
                    out.append(dict(level="error", var=sw_name, msg=f"median radiation at night {n_med:.0f} W m-2 "
                                    "- time shift or unit error", night_median=n_med))
            over = float(((sw > 1.3 * pot + 50) & sw.notna()).mean())
            if over > 0.01:
                out.append(dict(level="warning", var=sw_name, msg=f"{over:.1%} of half-hours above 1.3 x potential "
                                "radiation", share=over))

    # tower vs ERA5 air temperature, soil vs air temperature
    if a.has(ta_name):
        ta = a.series(ta_name)
        ta = ta - K0 if np.nanmedian(ta) > 150 else ta
        if a.has("t2m_era5"):
            te = a.site_series("t2m_era5")
            te = te - K0 if np.nanmedian(te) > 150 else te
            ok = ta.notna() & te.notna()
            if ok.sum() > 100:
                d, r = float((ta - te)[ok].mean()), float(np.corrcoef(ta[ok], te[ok])[0, 1])
                lvl = "error" if (abs(d) > 8 or r < 0.5) else "warning" if (abs(d) > 4 or r < 0.8) else "info"
                out.append(dict(level=lvl, var=ta_name, msg=f"tower vs ERA5 air temperature: bias {d:+.1f} K, r {r:.2f}",
                                bias=d, r=r))
        for sv in ("stl1_era5", "stl2_era5"):
            if a.has(sv):
                ts = a.series(sv)
                ts = ts - K0 if np.nanmedian(ts) > 150 else ts
                d = float(np.nanmean(ts) - np.nanmean(ta))
                if abs(d) > 12:
                    out.append(dict(level="warning", var=sv, msg=f"mean soil - air temperature {d:+.1f} K", diff=d))

    # energy balance closure (daytime)
    if all(a.has(v) for v in ("LE_F_MDS", "H_F_MDS", "NETRAD")) and a.has(sw_name):
        le, h, rn, sw = (a.series(v) for v in ("LE_F_MDS", "H_F_MDS", "NETRAD", sw_name))
        g = a.series("G_F_MDS").fillna(0) if a.has("G_F_MDS") else 0.0
        day = (sw > 200) & le.notna() & h.notna() & rn.notna()
        if day.sum() > 200:
            cr = float((le + h)[day].sum() / (rn - g)[day].sum())
            if not 0.4 <= cr <= 1.3:
                out.append(dict(level="warning", var="LE_F_MDS", msg=f"energy balance closure {cr:.2f} "
                                "(expected 0.6-1.1) - units or sign of LE / H / NETRAD?", closure=cr))
    if a.has("LE_RANDUNC") and a.has("LE_F_MDS"):
        r = float(np.nanmedian(a.series("LE_RANDUNC")) / max(np.nanmedian(np.abs(a.series("LE_F_MDS"))), 1.0))
        if r > 2:
            out.append(dict(level="warning", var="LE_RANDUNC", msg=f"LE_RANDUNC / |LE| median {r:.1f}", ratio=r))

    # NEE: no systematic uptake at night, less uptake at night than by day
    if a.has("NEE_VUT_REF") and a.has("SW_IN_POT"):
        nee, pot = a.series("NEE_VUT_REF"), a.series("SW_IN_POT")
        n_night, n_day = float(np.nanmedian(nee[pot <= 0])), float(np.nanmedian(nee[pot > 0]))
        if n_night < -1.0:
            out.append(dict(level="error", var="NEE_VUT_REF", msg=f"median night-time NEE {n_night:.2f} "
                            "(uptake at night) - sign convention or time shift?", night_median=n_night))
        if n_day > n_night + 0.5:
            out.append(dict(level="warning", var="NEE_VUT_REF", msg=f"median NEE higher by day ({n_day:.2f}) than "
                            f"at night ({n_night:.2f})", day_median=n_day, night_median=n_night))


# Pixels at the edge of the domain can be cut by the regridding (fractions < 1 there are expected).
# The check finds the narrowest edge frame outside of which all pixels sum to 1: if that frame is at
# most LC_EDGE_MAX pixels wide, the gaps are an edge effect (info); otherwise they are real gaps inside
# the image (warning).
LC_EDGE_MAX = 5
LC_EDGE_TOL = 0.005      # share of low pixels still allowed inside the LC_EDGE_MAX frame


def _check_land_cover(ds, out):
    if "land_cover_map" not in ds.variables:
        return
    lc = ds["land_cover_map"]
    cdim = [d for d in lc.dims if d not in ("x", "y", "lat", "lon")]
    if not cdim:
        return
    tot = lc.sum(cdim[0], skipna=True)
    if tot.ndim != 2:
        return
    t = np.asarray(tot.values, dtype=np.float64)
    ny, nx = t.shape
    yy, xx = np.indices(t.shape)
    dist = np.minimum.reduce([yy, xx, ny - 1 - yy, nx - 1 - xx])      # pixels to the nearest edge
    low, high = t < 0.99, t > 1.01
    if not (low.any() or high.any()):
        return
    if high.mean() > 0.001:
        out.append(dict(level="error" if high.mean() > 0.05 else "warning", var="land_cover_map",
                        msg=f"land-cover fractions sum to > 1.01 in {high.mean():.1%} of pixels", share_high=float(high.mean())))
    if not low.any():
        return
    # share of low pixels inside a frame of k pixels, k = 0, 1, ...
    k_max = int(dist.max())
    inner_share = [float(low[dist >= k].mean()) for k in range(k_max + 1)]
    k_needed = next((k for k, sh in enumerate(inner_share) if sh <= 0.001), None)
    profile = ", ".join(f"{k}: {inner_share[k]:.1%}" for k in range(min(LC_EDGE_MAX + 2, k_max + 1)))
    kk = min(LC_EDGE_MAX, k_max)
    # edge effect: the share falls with the frame width and (almost) vanishes within LC_EDGE_MAX pixels
    falling = all(inner_share[i + 1] <= inner_share[i] + 1e-4 for i in range(kk))
    if falling and inner_share[kk] <= LC_EDGE_TOL:
        out.append(dict(level="info", var="land_cover_map",
                        msg=f"{low.mean():.1%} of pixels sum to < 0.99, concentrated at the image edge "
                            f"({profile}) - regridding edge effect, ignored", k_needed=k_needed,
                        share_low=float(low.mean())))
        return
    k = LC_EDGE_MAX
    out.append(dict(level="warning", var="land_cover_map",
                    msg=f"land-cover fractions do not sum to 1 inside the image: {inner_share[k]:.1%} of pixels more than "
                        f"{k} pixels from the edge sum to < 0.99 (share of low pixels inside a k-pixel frame: {profile}"
                        + (f"; a {k_needed}-pixel frame would hold all of them - wide edge effect or gaps along the edge"
                           if k_needed is not None else "; gaps also deep inside the image") + ") - "
                        f"not explained by the edge; the model assigns the missing weight to evergreen forest, "
                        f"rerun the pipeline with --overwrite-landcover-cache for correct fractions",
                    share_low=inner_share[k], k_needed=k_needed, min_sum=float(np.nanmin(t[dist >= k]))))


def _check_sentinels(a, names, out):
    for v in names:
        if a.has(v):
            x = a.series(v)
            s = float((x <= -9990).mean())
            if s > 0 and a.has_time(v):
                sent = (x <= -9990).values
                if v in TOWER_INPUTS and s < 0.999 and a.series(v).ndim == 1:
                    a.set_series(v, _interp_at(x.mask(sent), sent).values)
                    msg = f"{s:.1%} values = -9999 interpolated in time (network input, never NaN)"
                else:
                    a.mask(v, sent)
                    msg = "not available (all values -9999)" if s > 0.999 else f"{s:.1%} values = -9999 set to NaN"
                out.append(dict(level="info", var=v, msg=msg, share=s, changed=True))


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def _run(a, fix_names, ta_name, sw_name):
    out = []
    names = [v for v in fix_names if a.has(v)]
    _check_sentinels(a, names, out)
    for v in names:
        spec = VAR_SPECS[v]
        if "either" in spec:
            if not a.has_time(v) or a.times is None:
                continue
            spec = _check_either_units(a, v, spec, out)
            if spec is None:
                continue
        else:
            _check_units(a, v, spec, out)
        _check_valid(a, v, spec, out)
        _check_flat(a, v, spec, out)
    _check_vpd(a, out, ta_name)
    _cross_checks(a, out, ta_name, sw_name)
    return out


def check_dataset(ds):
    """Checks (and unambiguous corrections) on the site dataset, in place. Returns (ds, findings)."""
    a = _DatasetAdapter(ds)
    names = [v for v in VAR_SPECS if v not in ("TA_F", "SW_IN_F")]
    out = _run(a, names, "t2m", "ssrd")
    _check_land_cover(a.ds, out)
    return a.ds, out


def check_dataframe(df, lon=None, times=None):
    """The same checks on the FLUXNET tower DataFrame (FLUXNET names), in place. Returns (df, findings)."""
    a = _FrameAdapter(df, times=times, lon=lon)
    names = [v for v in VAR_SPECS if v not in ("t2m", "ssrd") and not v.endswith("_era5")
             and v not in ("nirv", "lswi")]
    out = _run(a, names, "TA_F", "SW_IN_F")
    return a.df, out


def format_findings(findings, levels=("error", "warning", "correction")):
    lines = []
    for f in findings:
        if f["level"] in levels:
            lines.append(f"[{f['level'].upper():10s}] {f['var']:22s} {f['msg']}")
    return "\n".join(lines)