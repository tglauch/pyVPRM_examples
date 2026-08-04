#!/usr/bin/env python
"""
Train (or load) the pyvprnn_v1 or pyvprnn_v2 pixel model for a site, run inference to
produce footprint-aggregated GPP/Reco predictions, and generate the
standard diagnostic plots: a multi-panel example, a DT/NT comparison, a
diurnal-cycle climatology, and pixel-level PDP/ICE curves near the tower.

Reads the same config.yaml as run_vprm_pipeline.py and expects to find
that pipeline's output (out.nc) under the same {land_cover_type}_{site}
directory - see site_base_path() in utils.py.

Which model architecture to train/evaluate is set by model.version in
config.yaml ("pyvprnn_v1" or "pyvprnn_v2") - see resolve_pyvprnn_model()
in utils.py for the dispatch.

Usage:
    python train_and_evaluate.py --config config.yaml
    python train_and_evaluate.py --config config.yaml --site GF-Guy --fold 0
"""

import os
import logging
import argparse

import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt

import tensorflow as tf

from utils import (
    site_base_path,
    footprint_weighted_sum,
    crop_dataset_by_radius,
    maybe_extend_syspath,
    build_pyvprnn_kwargs,
    resolve_pyvprnn_model,
    make_generator,
    build_fixed_footprint
)

logger = logging.getLogger("train_and_evaluate")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    p.add_argument("--site", default=None, help="Override site.id from config")
    p.add_argument("--model", default=None, choices=["pyvprnn_v1", "pyvprnn_v2"],
                    help="Override model.version from config")
    p.add_argument("--t-start", default=None, help="Restrict training data to this start date (YYYY-MM-DD) onward")
    p.add_argument("--t-stop", default=None, help="Restrict training data up to this stop date (YYYY-MM-DD)")
    p.add_argument("--kfold", type=int, default=None, help="Override training.kfold")
    p.add_argument("--fold", type=int, default=None, help="Override evaluation.fold")
    p.add_argument("--make-plots", action="store_true", help="Save diagnostic plots")
    p.add_argument("--debug-numerics", action="store_true",
                    help="Override training.enable_check_numerics=true (slow - NaN/Inf checking on every op)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def load_config(path, args):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if args.site:
        cfg["site"]["id"] = args.site
    if args.model:
        cfg["model"]["version"] = args.model
    if args.kfold:
        cfg["training"]["kfold"] = args.kfold
    if args.fold is not None:
        cfg["evaluation"]["fold"] = args.fold
    if args.make_plots:
        cfg["plotting"]["enabled"] = True
    if args.debug_numerics:
        cfg["training"]["enable_check_numerics"] = True

    return cfg


def setup_determinism(cfg):
    """Fixed seed + deterministic ops, so repeated runs on the same data are reproducible."""
    train_cfg = cfg["training"]
    tf.keras.utils.set_random_seed(train_cfg["random_seed"])
    tf.config.experimental.enable_op_determinism()
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    if train_cfg["enable_check_numerics"]:
        logger.warning("enable_check_numerics is on - training will be noticeably slower.")
        tf.debugging.enable_check_numerics()


# ---------------------------------------------------------------------------
# Dataset loading / preparation
# ---------------------------------------------------------------------------

def load_training_dataset(base_path, cfg, t_start=None, t_stop=None, model="pyvprnn_v1"):
    pyvprnn_cls, _, _ = resolve_pyvprnn_model(model)

    ds_path = os.path.join(base_path, cfg["paths"]["training_data_filename"])
    ds = xr.open_dataset(ds_path)

    if t_start is not None or t_stop is not None:
        # slice(None, x) / slice(x, None) are open-ended on whichever side
        # is None, so this works whether one or both bounds are given.
        ds = ds.sel(
            datetime_utc=slice(t_start, t_stop),
            t=slice(t_start, t_stop),
        )

    # The satellite indices come out of the Kalman gap-filling step as
    # float64; cast back to float32 to match what the model was built for
    # and to roughly halve memory use.
    for var in ("evi", "lswi", "ndre", "nirv", "evi2"):
        if var in ds:
            ds[var] = ds[var].astype("float32")

    pyvprnn_inst = pyvprnn_cls(**build_pyvprnn_kwargs(cfg, model_name=model))
    pyvprnn_inst.set_ds(ds)
    fill_missing_jointunc(pyvprnn_inst.ds)
    return pyvprnn_inst


def fill_missing_jointunc(ds, missing_value=-9999.0):
    """Replace FLUXNET's missing-value sentinel in NEE_VUT_REF_JOINTUNC with the site mean uncertainty."""
    data = ds["NEE_VUT_REF_JOINTUNC"].data
    valid = data != missing_value
    if not valid.any():
        raise ValueError("NEE_VUT_REF_JOINTUNC has no valid (non-sentinel) values to average.")
    ds["NEE_VUT_REF_JOINTUNC"].data = np.where(data == missing_value, data[valid].mean(), data)


# ---------------------------------------------------------------------------
# Training / loading
# ---------------------------------------------------------------------------

def train_params_from_config(cfg):
    train_cfg = cfg["training"]
    return {
        "batch_size": train_cfg["batch_size"],
        "epochs": train_cfg["epochs"],
        "max_runtime_in_seconds": train_cfg["max_runtime_hours"] * 3600,
        "patience": train_cfg["patience"],
        "plateau_patience": train_cfg["plateau_patience"],
        "learning rate": train_cfg["learning_rate"],
        "workers": train_cfg["workers"],
        "multiprocessing": train_cfg["multiprocessing"],
        "max_queue_size": train_cfg["max_queue_size"],
        "loss": train_cfg["loss"],
    }


def train_or_load_folds(pyvprnn_inst, outpath, cfg):
    """
    For each CV fold: load the model if it already exists at model_path,
    otherwise train it from scratch. Returns {fold_index: model_path}.
    """
    train_cfg = cfg["training"]
    model_paths = {}

    for k in range(train_cfg["kfold"]):
        history_path = os.path.join(outpath, f"training_history_{k}.csv")
        model_path = os.path.join(outpath, f"pixel_model_{k}.keras")
        model_paths[k] = model_path

        if os.path.exists(model_path):
            logger.info("Fold %d: loading existing model at %s", k, model_path)
            pyvprnn_inst.load_model(model_path)
        else:
            logger.info("Fold %d: training a new model", k)
            open(history_path, "w").close()
            pyvprnn_inst.train(
                model_path,
                save_path_history=history_path,
                cv_fold=k,
                train_params=train_params_from_config(cfg),
            )

    return model_paths


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(pyvprnn_inst, cfg, ds_source=None, model="pyvprnn_v1", fixed_footprint=None):
    """
    Run the currently-loaded model over every timestep in ds_source (or
    pyvprnn_inst.ds_cropped if not given), aggregating each pixel-level
    GPP/Reco map into a footprint-weighted flux per timestep.
 
    If fixed_footprint is given (see build_fixed_footprint() in utils.py),
    ALSO aggregates the same raw per-pixel maps using that static footprint
    instead of the real time-varying one - this is a second aggregation of
    the exact same model output, not a second inference pass, since the
    footprint never feeds the model itself (it only post-processes the
    predicted maps). Useful for testing how much of the footprint-weighted
    output depends on the true, time-varying footprint shape versus a
    naive fixed window around the tower.
 
    Returns (times_sorted, results, example_batch) where results is a dict:
        results["dynamic"] = (gpp_sorted, reco_sorted)          # always present
        results["fixed"]   = (gpp_sorted, reco_sorted)          # only if fixed_footprint given
    example_batch is a fixed, explicitly-chosen batch (not "whatever batch
    happened to run last") suitable for the multi-panel diagnostic plot.
    """
    ds_cropped = ds_source if ds_source is not None else pyvprnn_inst.ds_cropped
    eval_cfg = cfg["evaluation"]
    _, _, is_lagged = resolve_pyvprnn_model(model)
 
    gen = make_generator(
        pyvprnn_inst, is_lagged,
        batch_size=eval_cfg["inference_batch_size"],
        times=pyvprnn_inst.common_times,
        shuffle=True,
        target="NEE_VUT_REF",
        unc="NEE_VUT_REF_JOINTUNC",
    )
    # NOTE: make_generator reads pyvprnn_inst.ds_cropped internally, not
    # ds_cropped/ds_source above - if you need to run inference over a
    # dataset other than the instance's own ds_cropped, this needs
    # extending (not currently used anywhere with a non-None ds_source).
 
    times = pyvprnn_inst.common_times[gen.indexes]
 
    results_gpp, results_reco = [], []
    results_gpp_fixed, results_reco_fixed = [], []
    example_batch = None
 
    for i in range(len(gen)):
        x, _ = gen[i]
        batch_times = gen.get_batch_times(i)
 
        if is_lagged:
            Xsat, Xstatic, Xmet, Xsw_in_pot, Xmet_lagged, footprint, flux_mask = x
            predict = pyvprnn_inst.pixel_model.predict_on_batch([Xsat, Xstatic, Xmet, Xsw_in_pot, Xmet_lagged, flux_mask])
        else:
            Xsat, Xstatic, Xmet, Xsw_in_pot, footprint, flux_mask = x
            predict = pyvprnn_inst.pixel_model.predict_on_batch([Xsat, Xstatic, Xmet, Xsw_in_pot, flux_mask])
 
        gpp, reco = predict[0].squeeze(), predict[1].squeeze()
 
        results_gpp.append(footprint_weighted_sum(gpp, footprint))
        results_reco.append(footprint_weighted_sum(reco, footprint))
 
        if fixed_footprint is not None:
            # fixed_footprint is (H, W), broadcasts against gpp/reco's
            # (batch, H, W) via the elementwise multiply inside
            # footprint_weighted_sum - same fixed weights applied to every
            # sample in the batch, no tiling needed.
            results_gpp_fixed.append(footprint_weighted_sum(gpp, fixed_footprint[None, ...]))
            results_reco_fixed.append(footprint_weighted_sum(reco, fixed_footprint[None, ...]))
 
        if i == eval_cfg["example_batch_index"]:
            example_batch = {"Xsat": Xsat, "batch_times": batch_times, "predict": predict}
 
    if example_batch is None:
        logger.warning("evaluation.example_batch_index=%d is out of range - using the last batch instead.",
                        eval_cfg["example_batch_index"])
        example_batch = {"Xsat": Xsat, "batch_times": batch_times, "predict": predict}
 
    sorted_idx = np.argsort(times)
    times_sorted = times[sorted_idx]
 
    results = {
        "dynamic": (
            np.concatenate(results_gpp)[sorted_idx],
            np.concatenate(results_reco)[sorted_idx],
        )
    }
    if fixed_footprint is not None:
        results["fixed"] = (
            np.concatenate(results_gpp_fixed)[sorted_idx],
            np.concatenate(results_reco_fixed)[sorted_idx],
        )
 
    return times_sorted, results, example_batch


def assign_predictions(ds_cropped, times_sorted, gpp_sorted, reco_sorted, model="pyvprnn_v1"):
    gpp_da = xr.DataArray(gpp_sorted, coords={"t": times_sorted}, dims=["t"]).reindex(
        t=ds_cropped["t"].values, fill_value=np.nan
    )
    reco_da = xr.DataArray(reco_sorted, coords={"t": times_sorted}, dims=["t"]).reindex(
        t=ds_cropped["t"].values, fill_value=np.nan
    )
    return ds_cropped.assign(**{
        f"{model}_gpp": gpp_da,
        f"{model}_reco": reco_da,
    })

def save_cropped_dataset(ds_cropped, out_path):
    coords_to_drop = [c for c in ds_cropped.coords if c not in ds_cropped.dims and c != "days_since_t0"]
    to_save = ds_cropped.reset_coords(coords_to_drop, drop=True)
    for v in to_save.variables.values():
        v.encoding = {}
    to_save.to_netcdf(out_path)
    logger.info("Wrote cropped prediction dataset to %s", out_path)
    return to_save


# ---------------------------------------------------------------------------
# Diurnal cycle plot
# ---------------------------------------------------------------------------

def _prepare_diurnal_dataset(ds, cfg, model="pyvprnn_v1"):
    ds = ds.assign_coords(datetime_t=("t", ds["datetime_utc"].data))
    ds = ds.swap_dims({"t": "datetime_t"}).drop_vars("datetime_utc").rename({"datetime_t": "datetime_utc"})

    ds[f"{model}_NEE"] = -ds[f"{model}_gpp"] + ds[f"{model}_reco"]
    if "GPP_DT_VUT_REF" in ds.keys():
        ds["NEE_DT_VUT_REF"] = -ds["GPP_DT_VUT_REF"] + ds["RECO_DT_VUT_REF"]
    if "GPP_NT_VUT_REF" in ds.keys():
        ds["NEE_NT_VUT_REF"] = -ds["GPP_NT_VUT_REF"] + ds["RECO_NT_VUT_REF"]

    time_coord = "datetime_utc"
    ds = ds.assign_coords(
        year=(time_coord, ds[time_coord].dt.year.data),
        month=(time_coord, ds[time_coord].dt.month.data),
        hour=(time_coord, ds[time_coord].dt.hour.data),
    )

    variables = [f"{model}_gpp", f"{model}_reco", f"{model}_NEE",
                 "GPP_DT_VUT_REF", "RECO_DT_VUT_REF",
                 "GPP_NT_VUT_REF", "RECO_NT_VUT_REF",
                 "NEE_VUT_REF", "NEE_DT_VUT_REF"]
    ref_var = next(v for v in variables if v in ds)

    min_obs = cfg["evaluation"]["diurnal_min_valid_days"] * cfg["evaluation"]["diurnal_obs_per_day"]
    valid_counts = ds[ref_var].groupby("year").count(dim=time_coord)
    valid_years = valid_counts["year"].where(valid_counts >= min_obs, drop=True).values
    ds = ds.sel({time_coord: ds["year"].isin(valid_years)})

    diurnal_ds = xr.Dataset()
    for var in variables:
        if var not in ds:
            logger.info("Variable %s not found - skipping in diurnal cycle plot.", var)
            continue
        data = -ds[var] if "gpp" in var.lower() else ds[var]
        diurnal_ds[var] = data.groupby("year").map(
            lambda yg: yg.groupby("month").map(lambda mg: mg.groupby("hour").mean(dim=time_coord))
        )

    spatial_dims = [d for d in diurnal_ds.dims if d not in ("year", "month", "hour")]
    return diurnal_ds.mean(dim=spatial_dims) if spatial_dims else diurnal_ds


def make_diurnal_cycle_plot(ds_cropped, cfg, site, out_path, model="pyvprnn_v1"):
    from pyVPRM.lib.fancy_plot import figsize

    plot_ds = _prepare_diurnal_dataset(ds_cropped, cfg, model=model)

    colors = {
        f"{model}_gpp": "#7570b3", f"{model}_reco": "#7570b3",
        "GPP_DT_VUT_REF": "#1b9e77", "RECO_DT_VUT_REF": "#1b9e77",
        "NEE_DT_VUT_REF": "#1b9e77", f"{model}_NEE": "#7570b3",
        "GPP_NT_VUT_REF": "#d95f02", "RECO_NT_VUT_REF": "#d95f02",
        "NEE_VUT_REF": "k",
    }

    years = sorted(plot_ds["year"].data)
    n_months = 12
    fs = figsize(1.3, 0.1)
    fig, axes = plt.subplots(
        nrows=len(years), ncols=n_months, figsize=(fs[0], len(years) * fs[1]),
        sharey=True, sharex=True, gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )
    # plt.subplots squeezes a single row/col to 1D already, so a plain
    # flatten() here always yields a consistent 1D, row-major (year, month)
    # ordering regardless of len(years) - no separate ndim branching needed.
    axes = np.atleast_1d(axes).flatten()

    for i, year in enumerate(years):
        year_label_ax = axes[i * n_months + (n_months - 1)]
        year_label_ax.text(1.06, 0.5, str(year), transform=year_label_ax.transAxes,
                            rotation=270, va="center", ha="left")

        months_available = set(plot_ds.sel(year=year)["month"].values.tolist())
        for m in range(1, n_months + 1):
            ax = axes[i * n_months + m - 1]
            ax.set_xticklabels([])
            if m == 1:
                ax.set_yticks([-20, 0])
            if m not in months_available:
                continue

            for var in plot_ds.data_vars:
                vals = plot_ds[var].sel(year=year, month=m)
                if not np.isfinite(vals).any():
                    continue
                ax.plot(plot_ds["hour"], vals, ls="dashed" if "NEE" in var else "-",
                         lw=0.7, color=colors.get(var))

    # Legend entries limited to variables actually plotted above (the
    # original included an "NT" entry even though NT variables were
    # commented out of `variables` and never drawn).
    axes[0].plot([], [], label=model.replace('pyvprnn_', 'pyVPRNN '), color="#7570b3")
    axes[0].plot([], [], label="DT", color="#1b9e77")
    axes[0].plot([], [], label="NT", color="#d95f02")
    axes[0].plot([], [], label="Obs.", color="k")
    handles, labels = axes[0].get_legend_handles_labels()
    top_ax = axes[6] if len(axes) > 6 else axes[0]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
               bbox_transform=top_ax.transAxes, ncol=len(labels), frameon=False)

    fig.text(0.04, 0.5, r"Flux [$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$]", va="center", rotation="vertical")
    fig.savefig(out_path.format(site=site), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diurnal cycle plot to %s", out_path.format(site=site))


# ---------------------------------------------------------------------------
# PDP / ICE
# ---------------------------------------------------------------------------

def make_pdp_ice_plot(ds_cropped_path, cfg, site, model_path, out_path, model="pyvprnn_v1"):
    from pyVPRM.lib.fancy_plot import figsize
    import plotting_functions

    pyvprnn_cls, _, is_lagged = resolve_pyvprnn_model(model)

    eval_cfg = cfg["evaluation"]
    summary = {}

    pyvprnn_inst_pdp = pyvprnn_cls(**build_pyvprnn_kwargs(cfg, model_name=model))
    with xr.open_dataset(ds_cropped_path) as ds:
        ds = ds.rio.write_crs(ds.attrs["crs"])
        cropped = crop_dataset_by_radius(
            ds, (ds.attrs["site_lon"], ds.attrs["site_lat"]), eval_cfg["pdp_crop_radius_km"]
        )
        pyvprnn_inst_pdp.set_ds(cropped)

    pyvprnn_inst_pdp.load_model(model_path)
    pyvprnn_inst_pdp.ds.compute()
    pyvprnn_inst_pdp.prepare_base_dataset(chunks_t=None, nee_qc_flags=cfg["training"]["nee_qc_flags"])
    pyvprnn_inst_pdp.split_train_val_test(k=1, val_frac=0.0, test_frac=0.0)

    gen = make_generator(pyvprnn_inst_pdp, is_lagged, batch_size=50000, times=pyvprnn_inst_pdp.common_times, shuffle=True)
    x, _ = gen[0]

    if is_lagged:
        Xsat, Xstatic, Xmet, Xsw_in_pot, Xmet_lagged, footprint, flux_mask = x
    else:
        Xsat, Xstatic, Xmet, Xsw_in_pot, footprint, flux_mask = x
        Xmet_lagged = None  
        
    if is_lagged:
        fig = plt.figure(figsize=figsize(0.55, ratio=0.4))
        ax1 = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    else:
        fig = plt.figure(figsize=figsize(1.1, ratio=0.4))
        ax1 = fig.add_axes((0.0, 0.0, 0.4, 1.0))
        ax2 = fig.add_axes((0.6, 0.0, 0.4, 1.0))

    _, t_values, ice_curves, gpp_pdp, _ = plotting_functions.plot_ice_pdp_gpp(
        pyvprnn_inst_pdp, Xmet, Xsat, Xstatic, Xsw_in_pot, flux_mask, opath="",
        ylabel=r"Normalized GPP [a.u.]", ax=ax1, fig=fig,
        met_stack_windowed=Xmet_lagged,
        show_band=True, show_ices=False, plot_pdp=True, add_colorbar=False)
    
    summary["gpp_t_values"], summary["gpp_ice_curves"], summary["gpp_pdp"] = t_values, ice_curves, gpp_pdp

    if not is_lagged:
        _, t_values, ice_curves, resp_pdp, _ = plotting_functions.plot_ice_pdp_resp(
            pyvprnn_inst_pdp, Xmet, Xsat, Xstatic, Xsw_in_pot, flux_mask, opath="",
            ylabel=r"Respiration [$\mu$molCO$_{2}$$\,\,$m$^{-2}$s$^{-1}$]", ax=ax2, fig=fig,
        )
        summary["reco_t_values"], summary["reco_ice_curves"], summary["reco_pdp"] = t_values, ice_curves, resp_pdp
    else:
        summary["reco_t_values"] = summary["reco_ice_curves"] = summary["reco_pdp"] = None

    fig.savefig(out_path.format(site=site), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved PDP/ICE plot to %s (Reco PDP %s)", out_path.format(site=site), "skipped" if is_lagged else "included")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)

    cfg = load_config(args.config, args)
    maybe_extend_syspath(cfg)
    setup_determinism(cfg)

    import socket
    logger.info("Running on %s (pid %d)", socket.gethostname(), os.getpid())
    logger.info("GPUs: %s", tf.config.list_physical_devices("GPU"))

    site = cfg["site"]["id"]
    model = cfg["model"]["version"]

    # NOTE: land_cover_type isn't known until the pipeline loads the flux
    # tower, so unlike run_vprm_pipeline.py we can't derive base_path from
    # scratch here - the directory must already exist from that run. If it
    # doesn't, run_vprm_pipeline.py first.
    candidates = [
        p for p in os.listdir(cfg["paths"]["output_base_dir"])
        if p.endswith(f"_{site}") and os.path.isdir(os.path.join(cfg["paths"]["output_base_dir"], p))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No {{land_cover_type}}_{site} directory found under {cfg['paths']['output_base_dir']} - "
            f"run run_vprm_pipeline.py for this site first."
        )
    if len(candidates) > 1:
        logger.warning("Multiple matching site directories found (%s) - using the first.", candidates)
    base_path = os.path.join(cfg["paths"]["output_base_dir"], candidates[0])

    range_tag = ""

    # Bring this back in asap!!!!!
    # if args.t_start or args.t_stop:
    #     range_tag = "_" + "_".join(v for v in (args.t_start, args.t_stop) if v)
    # Tagged by model too, so a v1 and v2 run against the same site/range
    # land in separate directories instead of overwriting each other's
    # pixel_model_*.keras / training_history_*.csv.
    outpath = os.path.join(base_path, f"{model}{range_tag}")
    os.makedirs(outpath, exist_ok=True)
    figure_dir = os.path.join(base_path, 'figures')
    os.makedirs(figure_dir, exist_ok=True)

    # --- data ------------------------------------------------------------------
    pyvprnn_inst = load_training_dataset(base_path, cfg, t_start=args.t_start, t_stop=args.t_stop, model=model)
    pyvprnn_inst.prepare_base_dataset(nee_qc_flags=cfg["training"]["nee_qc_flags"])
    pyvprnn_inst.split_train_val_test(
        k=cfg["training"]["kfold"], val_frac=cfg["training"]["val_frac"], test_frac=cfg["training"]["test_frac"]
    )

    fixed_footprint = build_fixed_footprint(pyvprnn_inst.ds_cropped, half_size=2)
    
    # --- train (or load) each fold ----------------------------------------------
    model_paths = train_or_load_folds(pyvprnn_inst, outpath, cfg)

    # --- inference ---------------------------------------------------------------
    eval_cfg = cfg["evaluation"]
    if eval_cfg["aggregate_folds"] and cfg["training"]["kfold"] > 1:
        all_times = []
        all_results = {"dynamic": ([], []), "fixed": ([], [])}
        for k, model_path in model_paths.items():
            pyvprnn_inst.load_model(model_path)
            t, results, example_batch = run_inference(pyvprnn_inst, cfg, model=model, fixed_footprint=fixed_footprint)
            all_times.append(t)
            for key in all_results:
                g, r = results[key]
                all_results[key][0].append(g)
                all_results[key][1].append(r)
 
        times_sorted = np.concatenate(all_times)
        order = np.argsort(times_sorted)
        times_sorted = times_sorted[order]
 
        final_results = {}
        for key, (gpp_list, reco_list) in all_results.items():
            final_results[key] = (
                np.concatenate(gpp_list)[order],
                np.concatenate(reco_list)[order],
            )
        results = final_results
        primary_model_path = model_paths[eval_cfg["fold"]]
    else:
        primary_model_path = model_paths[eval_cfg["fold"]]
        pyvprnn_inst.load_model(primary_model_path)
        times_sorted, results, example_batch = run_inference(pyvprnn_inst, cfg, model=model, fixed_footprint=fixed_footprint)
 
    # Dynamic-footprint predictions keep the existing model naming
    # (e.g. "pyvprnn_v1_gpp") - fixed-footprint predictions get a distinct
    # suffix so both live side by side in the same saved dataset without
    # colliding.
    gpp_dyn, reco_dyn = results["dynamic"]
    gpp_fixed, reco_fixed = results["fixed"]
 
    pyvprnn_inst.ds_cropped = assign_predictions(
        pyvprnn_inst.ds_cropped, times_sorted, gpp_dyn, reco_dyn, model=model
    )
    pyvprnn_inst.ds_cropped = assign_predictions(
        pyvprnn_inst.ds_cropped, times_sorted, gpp_fixed, reco_fixed, model=f"{model}_fixedfp"
    )

    ds_cropped_path = os.path.join(outpath, "ds_cropped.nc")
    saved_ds = save_cropped_dataset(pyvprnn_inst.ds_cropped, ds_cropped_path)

    if not cfg["plotting"]["enabled"]:
        return

    # --- diagnostic plots --------------------------------------------------------
    import plotting_functions

    plotting_functions.multi_panel_plot(
        pyvprnn_inst, example_batch["Xsat"], example_batch["batch_times"],
        example_batch["predict"], os.path.join(figure_dir, "multi_panels.png"),
    )
    plotting_functions.comparison_to_dt_nt(
        saved_ds,
        gpp_key='{}_gpp'.format(model),
        reco_key='{}_reco'.format(model),
        opath={"DT": os.path.join(figure_dir,
                                  "comparison_DT.png"),
               "NT": os.path.join(figure_dir,
                                  "comparison_NT.png")},
        nee_key="NEE_VUT_REF",
    )

    make_diurnal_cycle_plot(saved_ds, cfg, site, os.path.join(figure_dir,
                                                              "{site}_diurnal_cycle.png"), model=model)
    make_pdp_ice_plot(
        ds_cropped_path, cfg, site, primary_model_path,
        os.path.join(figure_dir, "{site}_pdp.png"), model=model,
    )


if __name__ == "__main__":
    main()