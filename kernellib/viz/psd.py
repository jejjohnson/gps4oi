import xarray as xr
import numpy as np
import logging
import matplotlib.pylab as plt
from scipy import interpolate
import hvplot.xarray
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle

def find_wavelength_05_crossing(filename):
    
    ds = xr.open_dataset(filename)
    y = 1./ds.wavenumber
    x = (1. - ds.psd_diff/ds.psd_ref)
    f = interpolate.interp1d(x, y)
    
    xnew = 0.5
    ynew = f(xnew)
    
    return ynew

def plot_psd_score(filename):
    
    ds = xr.open_dataset(filename)
    
    resolved_scale = find_wavelength_05_crossing(filename)
    
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(121)
    ax.invert_xaxis()
    plt.plot((1./ds.wavenumber), ds.psd_ref, label='reference', color='k')
    plt.plot((1./ds.wavenumber), ds.psd_study, label='reconstruction', color='lime')
    plt.xlabel('wavelength [km]')
    plt.ylabel('Power Spectral Density [m$^{2}$/cy/km]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.grid(which='both')
    
    ax = plt.subplot(122)
    ax.invert_xaxis()
    plt.plot((1./ds.wavenumber), (1. - ds.psd_diff/ds.psd_ref), color='k', lw=2)
    plt.xlabel('wavelength [km]')
    plt.ylabel('PSD Score [1. - PSD$_{err}$/PSD$_{ref}$]')
    plt.xscale('log')
    plt.hlines(y=0.5, 
              xmin=np.ma.min(np.ma.masked_invalid(1./ds.wavenumber)), 
              xmax=np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
              color='r',
              lw=0.5,
              ls='--')
    plt.vlines(x=resolved_scale, ymin=0, ymax=1, lw=0.5, color='g')
    ax.fill_betweenx((1. - ds.psd_diff/ds.psd_ref), 
                     resolved_scale, 
                     np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
                     color='green',
                     alpha=0.3, 
                     label=f'resolved scales \n $\lambda$ > {int(resolved_scale)}km')
    plt.legend(loc='best')
    plt.grid(which='both')
    
    logging.info(' ')
    logging.info(f'  Minimum spatial scale resolved = {int(resolved_scale)}km')
    
    plt.show()
    
    ds.close()
    
    return resolved_scale