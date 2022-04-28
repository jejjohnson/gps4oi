import xarray as xr
from pathlib import Path

def load_multiple_data(filenames):
    
    
    ds_obs = xr.open_mfdataset(
        filenames, combine="nested", concat_dim="time", parallel=True, preprocess=None
    ).sortby("time")
    
    return ds_obs

