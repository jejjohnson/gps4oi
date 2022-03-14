import xarray as xr
import pyinterp
from pathlib import PosixPath
from kernellib.features.config import load_test_aoi

def read_l3_data(
    ds,
    aoi=load_test_aoi()
):
    
    if isinstance(ds, list):
        ds = xr.open_mfdataset(ds, concat_dim ='time', combine='nested', parallel=True)
    elif isinstance(ds, str) or isinstance(ds, PosixPath):
        ds = xr.open_dataset(ds)
    elif isinstance(ds, xr.Dataset):
        pass
    else:
        raise ValueError(f"Unrecognized input")
        
    if aoi is not None:
        try:
            ds = ds.sel(time=slice(aoi.time_min, aoi.time_max), drop=True)
            ds = ds.where((ds["latitude"] >= aoi.lat_min) & (ds["latitude"] <= aoi.lat_max), drop=True)
            ds = ds.where((ds["longitude"] >= aoi.lon_min%360.) & (ds["longitude"] <= aoi.lon_max%360.), drop=True)
        except KeyError:
            ds = ds.sel(time=slice(aoi.time_min, aoi.time_max), drop=True)

            ds = ds.where((ds["lon"]%360. >= aoi.lon_min) & (ds["lon"]%360. <= aoi.lon_max), drop=True)
            ds = ds.where((ds["lat"] >= aoi.lat_min) & (ds["lat"] <= aoi.lat_max), drop=True)

    
    return ds


def l3_to_l4(ds, is_circle=True):
    
    x_axis = pyinterp.Axis(ds["lon"][:]%360., is_circle=is_circle)
    y_axis = pyinterp.Axis(ds["lat"][:])
    z_axis = pyinterp.TemporalAxis(ds["time"][:].data)
    
    var = ds['ssh'][:]
    var = var.transpose('lon', 'lat', 'time')

    # The undefined values must be set to nan.
    try:
        var[var.mask] = float("nan")
    except AttributeError:
        pass
    
    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.data)
    
    del ds
    
    return x_axis, y_axis, z_axis, grid