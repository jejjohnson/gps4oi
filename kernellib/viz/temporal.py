import xarray as xr
import hvplot.xarray

def plot_temporal_statistics(filename):
    
    ds1 = xr.open_dataset(filename, group='diff')
    ds2 = xr.open_dataset(filename, group='alongtrack')
    rmse_score = 1. - ds1['rms']/ds2['rms']
    
    rmse_score = rmse_score.dropna(dim='time').where(ds1['count'] > 10, drop=True)
    
    figure = rmse_score.hvplot.line(ylabel='RMSE SCORE', shared_axes=True, color='r') + ds1['count'].dropna(dim='time').hvplot.step(ylabel='#Obs.', shared_axes=True, color='grey')
    
    return figure.cols(1) 