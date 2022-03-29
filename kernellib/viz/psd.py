import xarray as xr
import wandb
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy import interpolate
import hvplot.xarray
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle


def find_wavelength_05_crossing(filename):

    ds = xr.open_dataset(filename)
    y = 1.0 / ds.wavenumber
    x = 1.0 - ds.psd_diff / ds.psd_ref
    f = interpolate.interp1d(x, y)

    xnew = 0.5
    ynew = f(xnew)

    return ynew


def plot_psd_score(filename, wandb_logger=None):

    ds = xr.open_dataset(filename)

    resolved_scale = find_wavelength_05_crossing(filename)
    resolved_scale = np.minimum(1_000, resolved_scale)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].invert_xaxis()
    ax[0].plot((1.0 / ds.wavenumber), ds.psd_ref, label="reference", color="k")
    ax[0].plot(
        (1.0 / ds.wavenumber), ds.psd_study, label="reconstruction", color="lime"
    )
    ax[0].set(
        xlabel="wavelength [km]",
        ylabel="Power Spectral Density [m$^{2}$/cy/km]",
        xscale="log",
        yscale="log",
    )
    ax[0].legend(loc="best")
    ax[0].grid(which="both")

    ax[1].invert_xaxis()
    ax[1].plot((1.0 / ds.wavenumber), (1.0 - ds.psd_diff / ds.psd_ref), color="k", lw=2)

    ax[1].set(
        xlabel="wavelength [km]",
        ylabel="PSD Score [1. - PSD$_{err}$/PSD$_{ref}$]",
        xscale="log",
    )
    ax[1].hlines(
        y=0.5,
        xmin=np.ma.min(np.ma.masked_invalid(1.0 / ds.wavenumber)),
        xmax=np.ma.max(np.ma.masked_invalid(1.0 / ds.wavenumber)),
        color="r",
        lw=0.5,
        ls="--",
    )
    ax[1].vlines(x=resolved_scale, ymin=0, ymax=1, lw=0.5, color="g")
    ax[1].fill_betweenx(
        (1.0 - ds.psd_diff / ds.psd_ref),
        resolved_scale,
        np.ma.max(np.ma.masked_invalid(1.0 / ds.wavenumber)),
        color="green",
        alpha=0.3,
        label=f"resolved scales \n $\lambda$ > {int(resolved_scale)}km",
    )
    ax[1].legend(loc="best")
    ax[1].grid(which="both")
    plt.tight_layout()

    logging.info(" ")
    logging.info(f"  Minimum spatial scale resolved = {int(resolved_scale)}km")

    if wandb_logger is not None:
        wandb_logger.log({"psd": wandb.Image(plt)})

    plt.show()
    ds.close()

    return resolved_scale
