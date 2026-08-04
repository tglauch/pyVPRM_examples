import numpy as np
import pandas as pd
from pyVPRM.lib.fancy_plot import *
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
import time

def multi_panel_plot(pyvprnn_inst, Xsat_test,
                     test_times,
                     predict, opath):

    test_times = pd.to_datetime(test_times)
    
    # define "midday" as closest to 12:00 local time
    midday_hour = 12
    hours_from_midday = np.abs(
        test_times.hour + test_times.minute / 60 + test_times.second / 3600 - midday_hour
    )
    
    ind = int(np.argmin(hours_from_midday))
    print("Chosen index:", ind)
    print("Chosen timestamp:", test_times[ind])
    
    # --- 2) get footprint summed over time ---------------------------------------
    summed_ffp = pyvprnn_inst.ds_cropped["ffp_footprint"].sum(dim="t")
    
    weighted_classes = (
        pyvprnn_inst.ds_cropped["land_cover_map"] * summed_ffp
    ).sum(dim=("y", "x"))
    
    dominant_vprm_class = int(weighted_classes.argmax(dim="vprm_classes").compute().item()) + 1
    print("Dominant VPRM class within footprint:", dominant_vprm_class)
    
    # extract that class map
    lc_map_dom = pyvprnn_inst.ds_cropped["land_cover_map"].sel(vprm_classes=dominant_vprm_class)
    
    # --- 4) pull arrays for plotting ---------------------------------------------
    evi = np.asarray(Xsat_test[ind, :, :, 1]).squeeze()
    lswi = np.asarray(Xsat_test[ind, :, :, 0]).squeeze()
    pred_gpp = np.asarray(predict[0][ind]).squeeze()
    pred_resp = np.asarray(predict[1][ind]).squeeze()
    lc_map_dom_np = np.asarray(lc_map_dom).squeeze()
    
    summed_ffp_np = np.asarray(summed_ffp).squeeze()
    
    def robust_limits(arr, qlow=2, qhigh=98, positive_only=False):
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if positive_only:
            a = a[a > 0]
        if a.size == 0:
            return None, None
        vmin, vmax = np.percentile(a, [qlow, qhigh])
        if np.isclose(vmin, vmax):
            vmin, vmax = a.min(), a.max()
        return float(vmin), float(vmax)
    
    # satellite vars
    evi_vmin, evi_vmax = robust_limits(evi)
    lswi_vmin, lswi_vmax = robust_limits(lswi)
    
    # predictions: usually nonnegative, so use 1-99% and floor at 0
    gpp_vmin, gpp_vmax = robust_limits(pred_gpp, qlow=1, qhigh=99, positive_only=False)
    resp_vmin, resp_vmax = robust_limits(pred_resp, qlow=1, qhigh=99, positive_only=False)
    gpp_vmin = max(0, gpp_vmin if gpp_vmin is not None else 0)
    resp_vmin = max(0, resp_vmin if resp_vmin is not None else 0)
    
    # land cover fraction usually 0-1
    lc_vmin, lc_vmax = 0, 1
    
    # footprint for optional overlay scaling
    ffp_vmin, ffp_vmax = robust_limits(summed_ffp_np, qlow=5, qhigh=99.5, positive_only=True)
    
    # --- 6) make single multi-panel plot -----------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    
    # Panel 1: dominant VPRM class fraction
    im0 = axes[0, 0].imshow(lc_map_dom_np, cmap="viridis", vmin=lc_vmin, vmax=lc_vmax)
    axes[0, 0].set_title(f"Land cover map\nDominant VPRM class in footprint = {dominant_vprm_class}")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Panel 2: EVI
    im1 = axes[0, 1].imshow(evi, cmap="Greens", vmin=evi_vmin, vmax=evi_vmax)
    axes[0, 1].set_title("Satellite channel 1 (EVI)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Panel 3: LSWI
    im2 = axes[0, 2].imshow(lswi, cmap="Reds", vmin=lswi_vmin, vmax=lswi_vmax)
    axes[0, 2].set_title("Satellite channel 0 (LSWI)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Panel 4: predicted GPP
    im3 = axes[1, 0].imshow(pred_gpp, cmap="Greens", vmin=gpp_vmin, vmax=gpp_vmax)
    axes[1, 0].set_title("Predicted GPP")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Panel 5: predicted respiration
    im4 = axes[1, 1].imshow(pred_resp, cmap="Purples", vmin=resp_vmin, vmax=resp_vmax)
    axes[1, 1].set_title("Predicted Respiration")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Panel 6: footprint itself
    im5 = axes[1, 2].imshow(summed_ffp_np, cmap="magma", vmin=ffp_vmin, vmax=ffp_vmax)
    axes[1, 2].set_title("Summed footprint")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle(f"Midday sample overview\nIndex={ind}, Time={test_times[ind]}", fontsize=16)
    fig.savefig(
         opath,
         dpi=300,
         bbox_inches="tight")
    plt.show()


def comparison_to_dt_nt(ds, opath=None, 
                        gpp_key='pyvprnn_v1_gpp',
                        reco_key='pyvprnn_v1_reco',
                        nee_key='NEE_VUT_REF'):
    
    gpp_sum_pred  = ds[gpp_key]
    reco_sum_pred = ds[reco_key]
    
    nee_sum_pred = -gpp_sum_pred + reco_sum_pred
    
    for part_method in ['DT', 'NT']:
        if f'GPP_{part_method}_VUT_REF' not in ds.keys():
            continue
        fig = plt.figure(figsize=figsize(1.4, ratio=0.2))
        # Helper function
        def add_hexbin_panel(ax, x, y, xlabel, ylabel):
            x = np.asarray(x)
            y = np.asarray(y)
    
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.all():
                n_dropped = int((~finite).sum())
                print(f"add_hexbin_panel: dropping {n_dropped} non-finite point(s) out of {len(x)} "
                      f"(likely pyvprnn_v2's lag-window gap or another data gap).")
                x, y = x[finite], y[finite]
        
            if len(x) < 2:
                print("add_hexbin_panel: fewer than 2 finite points remain - skipping this panel.")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                return None
        
            try:
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy)
                density = kde(xy)
        
                # Normalize density for plotting
                density_norm = (density - density.min()) / (density.max() - density.min())
        
                # Use density for alpha
                alpha_scaled = np.log1p(density * 100)
                alpha_scaled = alpha_scaled / alpha_scaled.max()
        
            except np.linalg.LinAlgError:
                print("Warning: KDE failed — falling back to constant values")
                alpha_default = 0.1
                alpha_scaled = np.full_like(x, alpha_default, dtype=float)
                density_norm = np.zeros_like(x)
        
            hb = ax.scatter(
                x,
                y,
                s=1,
                marker='o',
                alpha=alpha_scaled,
                c=density_norm,
                cmap='inferno',
                edgecolor='none'
            )
        
            lim = np.nanpercentile(np.concatenate([x, y]), 99)
            lim_min = np.nanpercentile(np.concatenate([x, y]), 1)
            ax.plot([lim_min, lim], [lim_min, lim], color='grey', alpha=0.5, linestyle='-.')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(lim_min, lim)
            ax.set_ylim(lim_min, lim)
        
            # R²
            r2 = r2_score(x, y)
            ax.text(
                0.05, 0.95,
                rf"$R^2 = {r2:.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
            return hb
                
        # -----------------------
        # Panel 1 — NEE
        # -----------------------
        ax = fig.add_axes((0.0, 0.0, 0.18, 1.0))
        mask_qc = ds['NEE_VUT_REF_QC'].values == 0
        
        x1 = ds[nee_key].values[mask_qc]
        y1 = nee_sum_pred.values[mask_qc]
        
        add_hexbin_panel(
            ax,
            x1,
            y1,
            r'NEE_VUT_REF',
            r'NEE$_{\text{predicted}}$')
        
        # -----------------------
        # Panel 2 — GPP
        # -----------------------
        ax2 = fig.add_axes((0.28, 0.0, 0.18, 1.0))
        
        x2 = ds[f'GPP_{part_method}_VUT_REF'].values[mask_qc]
        y2 = gpp_sum_pred.values[mask_qc]
        
        add_hexbin_panel(
            ax2,
            x2,
            y2,
            f'GPP_{part_method}_VUT_REF',
            r'GPP$_{\text{predicted}}$')
        
        # -----------------------
        # Panel 3 — RECO (day)
        # -----------------------
        ax3 = fig.add_axes((0.56, 0.0, 0.18, 1.0))
        
        mask_day = ds['ssrd'].values > 0
        
        x3 = ds[f'RECO_{part_method}_VUT_REF'].values[mask_day & mask_qc]
        y3 = reco_sum_pred[mask_day& mask_qc].values
        
        add_hexbin_panel(
            ax3,
            x3,
            y3,
            rf'RECO_{part_method}_VUT_REF [day]',
            r'R$_{\text{eco, predicted}}$'
        )
        
        # -----------------------
        # Panel 4 — RECO (night)
        # -----------------------
        ax4 = fig.add_axes((0.84, 0.0, 0.18, 1.0))
        
        mask_night = ds['ssrd'].values == 0
        
        x4 = ds[nee_key].values[mask_night& mask_qc]
        y4 = reco_sum_pred.values[mask_night& mask_qc]
        
        add_hexbin_panel(
            ax4,
            x4,
            y4,
            r'NEE_VUT_REF [night]',
            r'R$_{\text{eco, predicted}}$ [night]')
    
        ax5 = fig.add_axes((1.12, 0.0, 0.18, 1.0))  
        mask_night = ds['ssrd'].values == 0
        
        x5 = ds[nee_key].values[mask_night& mask_qc]
        y5 = ds[f'RECO_{part_method}_VUT_REF'].values[mask_night& mask_qc]
        
        add_hexbin_panel(
            ax5,
            x5,
            y5,
            r'NEE_VUT_REF [night]',
            rf'RECO_{part_method}_VUT_REF [night]')
    
        ax6 = fig.add_axes((1.4, 0.0, 0.18, 1.0))
        
        x6 = ds[nee_key].values[mask_qc]
        y6 = ds[f'RECO_{part_method}_VUT_REF'].values[mask_qc] - ds[f'GPP_{part_method}_VUT_REF'].values[mask_qc]
        
        add_hexbin_panel(
            ax6,
            x6,
            y6,
            r'NEE_VUT_REF',
            f'NEE_{part_method}_VUT_REF')

        if opath is not None:
            fig.savefig(
                opath[part_method],
                dpi=300,
                bbox_inches="tight")


def plot_ice_pdp_gpp(pyvprnn_inst,
                     Xmet_test,
                     Xsat_test,
                     Xsat_static_test,
                     Xsw_in_pot_test,
                     flux_mask_test,
                     opath=None,
                     ylabel=None,
                     met_stack_windowed=None,
                     normalize_ice_curves=True,
                     show_ices=True,
                     show_band=False,
                     show_band_color='gray',
                     plot_pdp=True,
                     ax=None,
                     fig=None,
                     rh_min_max=None,
                     add_colorbar=False):
    max_vprm_class = (pyvprnn_inst.ds_cropped['land_cover_map'] * pyvprnn_inst.ds_cropped['ffp_footprint']).sum(dim=['x', 'y', 't']).argmax() + 1
    print(max_vprm_class)
    summed = pyvprnn_inst.ds_cropped['ffp_footprint'].sum(dim='t')
    inds = summed.where((pyvprnn_inst.ds_cropped.sel({'vprm_classes':max_vprm_class})['land_cover_map']>0.99)).argmax(dim=("y","x"))
    Xmet_test = Xmet_test.squeeze()

    ssrd_idx = pyvprnn_inst.met_vars.index("ssrd")
    t2m_idx = pyvprnn_inst.met_vars.index("t2m")
    rh_idx = pyvprnn_inst.met_vars.index("RH_from_VDP")
    sx = int(inds['y'].values)
    sy = int(inds['x'].values)
    # Condition mask
    if rh_min_max is None:
        high_light_mask = (
            (Xmet_test[:,ssrd_idx] > np.nanpercentile(Xmet_test[:,ssrd_idx], 75)) & 
            (Xmet_test[:,ssrd_idx] < np.nanpercentile(Xmet_test[:,ssrd_idx], 85)) & 
            (Xsat_test[:,sx,sy,2] > np.nanpercentile(Xsat_test[:,sx,sy,2], 70))&
            (Xsat_test[:,sx,sy,2] < np.nanpercentile(Xsat_test[:,sx,sy,2], 80))&
            (Xmet_test[:, rh_idx] > np.nanpercentile(Xmet_test[:, rh_idx], 30)) &
            (Xmet_test[:, rh_idx] < np.nanpercentile(Xmet_test[:, rh_idx], 100)))
    else:
        high_light_mask = (
            (Xmet_test[:,ssrd_idx] > np.nanpercentile(Xmet_test[:,ssrd_idx], 75)) & 
            (Xmet_test[:,ssrd_idx] < np.nanpercentile(Xmet_test[:,ssrd_idx], 85)) & 
            (Xsat_test[:,sx,sy,2] > np.nanpercentile(Xsat_test[:,sx,sy,2], 70))&
            (Xsat_test[:,sx,sy,2] < np.nanpercentile(Xsat_test[:,sx,sy,2], 80))&
            (Xmet_test[:, rh_idx] > rh_min_max[0])& 
            (Xmet_test[:, rh_idx] < rh_min_max[1])) 
   #)
    print(len(Xmet_test[high_light_mask]))
    color_by_full = Xsat_test[:,sx,sy,2]  # shape: (n_samples,)
    
    t_values, ice_curves, gpp_pdp, subsample_idx = pyvprnn_inst.partial_dependence_preprocessed_ice(
        X_sat=Xsat_test[:, sx:sx+1, sy:sy+1, :],
        X_sat_static=Xsat_static_test[:, sx:sx+1, sy:sy+1, :],
        X_met=Xmet_test,
        X_mask=flux_mask_test[:, sx:sx+1, sy:sy+1],
        X_sw_in_pot=Xsw_in_pot_test,
        X_met_lagged=met_stack_windowed,
        feature_idx=t2m_idx,
        condition_mask=high_light_mask,
        normalize_ice_curves=normalize_ice_curves,
        n_points=80,
        subsample=100,
        output_var_idx=0,
        add_to_temp_range_min=4,
        add_to_temp_range_max=2,
    )
    
    # Subsample the color variable exactly the same way
    color_by = color_by_full[high_light_mask][subsample_idx]
    
    # Plot
    ax = pyvprnn_inst.plot_ice_pdp(
        f_values=t_values,
        ice=ice_curves,
        pdp=gpp_pdp,
        xlabel="Temperature (°C)",
        ylabel=ylabel,
        title='', #flux_tower_inst.site_name,
        color_var=color_by,
        out_path=opath,
        ax=ax,
        fig=fig,
        show_band_color=show_band_color,
        show_ices=show_ices,
        show_band=show_band,
        plot_pdp=plot_pdp,
        plot_colorbar=add_colorbar,
        cbar_label='NDRE')
    return ax, t_values, ice_curves, gpp_pdp, subsample_idx

def plot_ice_pdp_resp(pyvprnn_inst,
                     Xmet_test,
                     Xsat_test,
                     Xsat_static_test,
                     Xsw_in_pot_test,
                     flux_mask_test,
                     opath=None,
                     ylabel=None,
                     met_stack_windowed=None,
                     normalize_ice_curves=False,
                     add_colorbar=True,
                     ax=None,
                     fig=None):

    max_vprm_class = (pyvprnn_inst.ds_cropped['land_cover_map'] * pyvprnn_inst.ds_cropped['ffp_footprint']).sum(dim=['x', 'y', 't']).argmax() + 1
    summed = pyvprnn_inst.ds_cropped['ffp_footprint'].sum(dim='t')
    inds = summed.where((pyvprnn_inst.ds_cropped.sel({'vprm_classes':max_vprm_class})['land_cover_map']>0.99)).argmax(dim=("y","x"))
    Xmet_test = Xmet_test.squeeze()

    ssrd_idx = pyvprnn_inst.met_vars.index("ssrd")
    t2m_idx = pyvprnn_inst.met_vars.index("t2m")
    sx = int(inds['y'].values)
    sy = int(inds['x'].values)
    bins = np.linspace(0, 100, 10)
    vmin = np.nanmin(Xsat_test[:,sx,sy,2])
    vmax = np.nanmax(Xsat_test[:,sx,sy,2])
    # Condition mask
    all_t_values = []
    all_ice_curves = []
    all_gpp_pdp = []
    for i in range(len(bins)-1):
        high_light_mask = (
            (Xmet_test[:, 2] > np.nanpercentile(Xmet_test[:, 2], 20)) &
            (Xmet_test[:, 2] < np.nanpercentile(Xmet_test[:, 2], 90)) &
            (Xsat_test[:,sx,sy,2] > np.nanpercentile(Xsat_test[:,sx,sy,2], bins[i])) & 
            (Xsat_test[:,sx,sy,2] < np.nanpercentile(Xsat_test[:,sx,sy,2], bins[i+1])))
        #print(len(Xmet_test[high_light_mask]))
        color_by_full = Xsat_test[:,sx,sy,2]  # shape: (n_samples,)

        # Compute ICE + PDP
        t_values, ice_curves, gpp_pdp, subsample_idx = pyvprnn_inst.partial_dependence_preprocessed_ice(
            X_sat=Xsat_test[:, sx:sx+1, sy:sy+1, :],
            X_sat_static=Xsat_static_test[:, sx:sx+1, sy:sy+1, :],
            X_met=Xmet_test,
            X_mask=flux_mask_test[:, sx:sx+1, sy:sy+1],
            X_sw_in_pot=Xsw_in_pot_test,
            X_met_lagged=met_stack_windowed,
            feature_idx=t2m_idx,
            condition_mask=high_light_mask,
            normalize_ice_curves=normalize_ice_curves,
            n_points=80,
            subsample=int(100/len(bins)),
            output_var_idx=1,
            add_to_temp_range_min=0,
            add_to_temp_range_max=0)
        all_t_values.append(t_values)
        all_ice_curves.append(ice_curves)
        all_gpp_pdp.append(gpp_pdp)
        
        # Subsample the color variable exactly the same way
        # if i + 2 == len(bins):
        #     plot_colorbar=True
        # else:
        #     plot_colorbar= False
        color_by = color_by_full[high_light_mask][subsample_idx]
        
        ax = pyvprnn_inst.plot_ice_pdp(
            f_values=t_values,
            ice=ice_curves,
            pdp=gpp_pdp,
            xlabel="Temperature (°C)",
            ylabel=ylabel,
            title='', #flux_tower_inst.site_name,
            color_var=color_by,
            out_path=opath,
            ax=ax,
            fig=fig,
            vmin = vmin,
            vmax = vmax,
            plot_pdp=False,
            plot_colorbar=add_colorbar,
            cbar_label='NDRE')
    return ax, all_t_values, all_ice_curves, all_gpp_pdp, subsample_idx