#!/usr/bin/env python
"""
Site files: unit / consistency checks, derived variables, and the list of variables the model needs.

    prepare_site_dataset(ds)  ->  (ds, findings, ok)
        1. site_checks.check_dataset: units per month, impossible values, constant runs, VPD against
           e_s(TA), time stamps in UTC, cross-checks (see site_checks.py); unambiguous problems are
           corrected (original kept as <var>_orig), everything else reported
        2. derived variables from the CHECKED inputs: RH_from_VPD
        3. model inputs present (missing -> error)
        ok = no finding at level "error"

Used by run_vprm_pipeline.py before the dataset is written, and standalone to check / repair
existing site files without reprocessing them:

    python derived_variables.py /path/to/sites/*/out.nc               # check, repair, add, rewrite
    python derived_variables.py --check-only /path/to/sites/*/out.nc  # report only, files unchanged
    python derived_variables.py --report qc_sites.csv ...             # all findings as a table

Files are rewritten atomically (temporary file + rename) and only if something changed. Running
it again on a repaired file changes nothing. Exit code 1 if any file has errors.
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from site_checks import check_dataset, magnus_hpa, format_findings, K0

# Units of the derived relative humidity, as computed in pyvprnn's prepare_base_dataset
# (vpd_hpa_to_rh(...) / 100 -> fraction).
RH_UNITS = "fraction"

# Dataset variables the model (pyvprnn_v2) reads; missing -> it stops with an error.
MODEL_INPUTS = (
    "t2m", "ssrd", "RH_from_VPD", "swvl1_era5", "swvl2_era5", "sd_era5", "stl1_era5", "stl2_era5",
    "P_F", "SW_IN_POT", "lswi", "nirv", "land_cover_map",
    "NEE_VUT_REF", "NEE_VUT_REF_RANDUNC",
)
# Needed for the diffusion bound (training without it otherwise) and for the derived variables.
OPTIONAL_INPUTS = ("LE_F_MDS", "H_F_MDS", "LE_RANDUNC", "VPD_F", "WS_F", "USTAR", "PA_F")
# old names accepted for a variable (new name: old names)
ALIASES = {"RH_from_VPD": ("RH_from_VDP",)}


def _present(ds, name):
    return name in ds.variables or any(a in ds.variables for a in ALIASES.get(name, ()))


def relative_humidity(vpd_hpa, ta_c, units=RH_UNITS):
    """RH = 1 - VPD / e_s(T), clipped to [0, 1] (VPD in hPa, TA in degC)."""
    rh = np.clip(1.0 - np.asarray(vpd_hpa) / magnus_hpa(np.asarray(ta_c)), 0.0, 1.0)
    return (rh * 100.0 if units == "percent" else rh).astype("float32")


def add_derived_variables(ds, vpd_changed=False, overwrite=False):
    """RH_from_VPD from the checked VPD_F and tower t2m. An old misspelled RH_from_VDP is renamed
    (values unchanged) unless VPD_F was corrected in this run or overwrite is set - then RH is
    recomputed. Returns (ds, findings)."""
    out = []
    if "RH_from_VDP" in ds.variables and "RH_from_VPD" not in ds.variables and not (overwrite or vpd_changed):
        ds = ds.rename({"RH_from_VDP": "RH_from_VPD"})
        out.append(dict(level="correction", var="RH_from_VPD", msg="renamed from RH_from_VDP (values unchanged)"))
    if overwrite or vpd_changed or "RH_from_VPD" not in ds.variables:
        missing = [v for v in ("VPD_F", "t2m") if v not in ds.variables]
        if missing:
            out.append(dict(level="error", var="RH_from_VPD", msg=f"cannot be derived: {missing} missing"))
        else:
            ta = ds["t2m"].values.astype(np.float64)
            ta = ta - K0 if np.nanmedian(ta) > 150 else ta
            rh = relative_humidity(ds["VPD_F"].values, ta)
            ds["RH_from_VPD"] = (ds["VPD_F"].dims, rh,
                                 {"long_name": "relative humidity from tower VPD_F and TA_F",
                                  "units": "%" if RH_UNITS == "percent" else "1",
                                  "formula": "1 - VPD / e_s(TA), Magnus (6.1078 hPa, 17.27, 237.3)"})
            out.append(dict(level="correction", var="RH_from_VPD",
                            msg=f"computed (median {float(np.nanmedian(rh)):.2f}, missing {float(np.isnan(rh).mean()):.1%})"))
    if "RH_from_VDP" in ds.variables:
        ds = ds.drop_vars("RH_from_VDP")
        out.append(dict(level="correction", var="RH_from_VDP", msg="misspelled variable removed"))
    if "RH_from_VPD" in ds.variables:
        rh = ds["RH_from_VPD"].values
        scale = 100.0 if np.nanmedian(rh) > 1.5 else 1.0
        zero, sat = float(np.nanmean(rh <= 0.001 * scale)), float(np.nanmean(rh >= 0.999 * scale))
        if zero > 0.02 or sat > 0.3:
            out.append(dict(level="error" if zero > 0.2 else "warning", var="RH_from_VPD",
                            msg=f"{zero:.1%} at 0 and {sat:.1%} at saturation - VPD or TA still inconsistent?",
                            share_zero=zero, share_saturated=sat))
    return ds, out


def check_model_inputs(ds):
    """(missing required, missing optional) dataset variables."""
    missing = [v for v in MODEL_INPUTS if not _present(ds, v)]
    missing_opt = [v for v in OPTIONAL_INPUTS if not _present(ds, v)]
    return missing, missing_opt


def prepare_site_dataset(ds, overwrite=False):
    """Checks, corrections and derived variables. Returns (ds, findings, ok)."""
    vpd_before = ds["VPD_F"].values.copy() if "VPD_F" in ds.variables else None
    ds, findings = check_dataset(ds)
    vpd_changed = vpd_before is not None and not np.allclose(vpd_before, ds["VPD_F"].values, equal_nan=True)
    ta_changed = any(f["var"] == "t2m" and (f["level"] == "correction" or f["msg"].startswith("UNIT CHANGE"))
                     for f in findings)
    ds, f2 = add_derived_variables(ds, vpd_changed=vpd_changed or ta_changed, overwrite=overwrite)
    findings += f2
    # inputs present but mostly missing (after the checks)
    for v in dict.fromkeys(MODEL_INPUTS + OPTIONAL_INPUTS):
        if v in ds.variables and "datetime_utc" in ds[v].dims:
            x = ds[v]
            other = [d for d in x.dims if d != "datetime_utc"]
            x = x.mean(other, skipna=True) if other else x
            miss = float(np.mean(~np.isfinite(x.values) | (x.values <= -9990)))
            if miss > 0.999 and v in MODEL_INPUTS:
                findings.append(dict(level="error", var=v, msg="entirely missing - the model will not run"))
            elif miss > 0.5:
                why = {"NEE_VUT_REF_RANDUNC": " - training uses one constant sigma (mean of the valid values)",
                       "LE_RANDUNC": " - the diffusion bound applies only where it is available",
                       "LE_F_MDS": " - the diffusion bound applies only where it is available"}.get(v, "")
                findings.append(dict(level="warning", var=v, msg=f"{miss:.0%} missing{why}", missing=miss))
    missing, missing_opt = check_model_inputs(ds)
    if missing:
        findings.append(dict(level="error", var="model inputs", msg=f"missing - the model will not run: {missing}"))
    if missing_opt:
        findings.append(dict(level="warning", var="optional inputs",
                             msg=f"missing (diffusion bound / derived variables): {missing_opt}"))
    ok = not any(f["level"] == "error" for f in findings)
    ds.attrs["site_checks"] = json.dumps(findings, default=str)
    ds.attrs["site_checks_status"] = "ok" if ok else "errors"
    return ds, findings, ok


def _write(ds, path):
    for v in ds.variables.values():
        v.encoding = {}
    tmp = path + ".tmp"
    try:
        ds.to_netcdf(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def process_file(path, overwrite=False, check_only=False, verbose=False):
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    site = ds.attrs.get("site_name", ds.attrs.get("site", os.path.basename(os.path.dirname(path))))
    nvars_before = set(ds.variables)
    ds, findings, ok = prepare_site_dataset(ds, overwrite=overwrite)
    changed = any(f["level"] == "correction" or f.get("changed") for f in findings) \
        or set(ds.variables) != nvars_before
    print(f"\n{'=' * 100}\n{site}  ({path})  ->  {'OK' if ok else 'ERRORS'}")
    txt = format_findings(findings, levels=("error", "warning", "correction") + (("info",) if verbose else ()))
    print(txt if txt else "  no problems found")
    if changed and not check_only:
        _write(ds, path)
        print("  -> file rewritten (originals kept as <var>_orig)")
    elif changed:
        print("  -> corrections NOT written (--check-only)")
    return site, ok, findings


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--overwrite", action="store_true", help="recompute derived variables that already exist")
    p.add_argument("--check-only", action="store_true", help="report only, do not rewrite the files")
    p.add_argument("--report", default=None, help="write all findings of all files to this CSV")
    p.add_argument("--verbose", action="store_true", help="also print info lines (percentiles, -9999, ...)")
    a = p.parse_args()
    rows, status = [], []
    for f in a.files:
        try:
            site, ok, findings = process_file(f, a.overwrite, a.check_only, a.verbose)
        except Exception as e:                                  # one broken file must not stop the batch
            site, ok, findings = f, False, [dict(level="error", var="file", msg=f"{type(e).__name__}: {e}")]
            print(f"\n{f}: {type(e).__name__}: {e}")
        status.append((site, ok))
        rows += [dict(site=site, file=f, **{k: (v if k in ("level", "var", "msg") else None) for k, v in x.items()})
                 for x in findings]
    print(f"\n{'=' * 100}\n{sum(ok for _, ok in status)}/{len(status)} files without errors")
    bad = [s for s, ok in status if not ok]
    if bad:
        print("with errors: " + ", ".join(map(str, bad)))
    if a.report:
        pd.DataFrame(rows)[["site", "file", "level", "var", "msg"]].to_csv(a.report, index=False)
        print(f"findings written to {a.report}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()