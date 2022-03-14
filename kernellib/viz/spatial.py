import xarray as xr

def plot_spatial_statistics(filename):
    
    ds = xr.open_dataset(filename, group='diff')
#     try:
#         figure = ds['rmse'].hvplot.image(x='lon', y='lat', z='rmse', clabel='RMSE [m]', cmap='Reds', coastline=True)
#     except KeyError:
#         figure = ds['rmse'].hvplot.image(x='lon', y='lat', z='rmse', clabel='RMSE [m]', cmap='Reds')

    figure = ds['rmse'].hvplot.image(x='lon', y='lat', z='rmse', clabel='RMSE [m]', cmap='Reds')
        
    return figure
