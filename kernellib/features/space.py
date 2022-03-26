import xarray as xr

def correct_lon(ds, lon_min):

    if lon_min < 0:
        ds["longitude"] = xr.where(
            ds["longitude"] >= 180.0, ds["longitude"] - 360.0, ds["longitude"]
        )
    return ds