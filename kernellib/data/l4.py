import xarray as xr
import pyinterp
from kernellib.features.config import load_test_aoi
from kernellib.data.l3 import read_l3_data, l3_to_l4


def read_l4_data(
    ds,
    aoi=load_test_aoi(),
    is_circle=True
):
    ds = read_l3_data(ds, aoi)
    
    x_axis, y_axis, z_axis, grid = l3_to_l4(ds, is_circle=is_circle)
#     if isinstance(ds, list):
#         ds = xr.open_mfdataset(ds, concat_dim ='time', combine='nested', parallel=True)
#     elif isinstance(ds, str):
#         ds = xr.open_dataset(ds)
#     elif isinstance(ds, xr.Dataset):
#         pass
#     else:
#         raise ValueError(f"Unrecognized input")
        
#     if aoi is not None:
#         ds = ds.sel(time=slice(aoi.time_min, aoi.time_max), drop=True)

#         ds = ds.where((ds["lon"]%360. >= aoi.lon_min) & (ds["lon"]%360. <= aoi.lon_max), drop=True)
#         ds = ds.where((ds["lat"] >= aoi.lat_min) & (ds["lat"] <= aoi.lat_max), drop=True)
    
#     x_axis = pyinterp.Axis(ds["lon"][:]%360., is_circle=is_circle)
#     y_axis = pyinterp.Axis(ds["lat"][:])
#     z_axis = pyinterp.TemporalAxis(ds["time"][:].data)
    
#     var = ds['ssh'][:]
#     var = var.transpose('lon', 'lat', 'time')

#     # The undefined values must be set to nan.
#     try:
#         var[var.mask] = float("nan")
#     except AttributeError:
#         pass
    
#     grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.data)
    
    del ds
    
    return x_axis, y_axis, z_axis, grid