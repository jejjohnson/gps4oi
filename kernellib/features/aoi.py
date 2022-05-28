from dataclasses import dataclass
import numpy as np
from kernellib.features.space import correct_lon


def aoi_factory(aoi: str):
    
    if aoi == "dc2021":
        return AOIParams()
    elif aoi == "dc2021sm":
        return AOIParams()
    else:
        raise ValueError(f"Unrecognized aoi params: {aoi}")

@dataclass
class AOIParams:
    lon_min = 295.0  # domain min longitude
    lon_max = 305.0  # domain max longitude
    lon_ext = 0.0
    lat_min = 33.0  # domain min latitude
    lat_max = 43.0  # domain max latitude
    lat_ext = 0.0
    time_min = np.datetime64("2017-01-01")  # domain min time
    # time_max = np.datetime64("2017-01-15")
    time_max = np.datetime64("2017-12-31")
    time_ext = 0.0
    dlon = 0.2  # zonal grid spatial step (in degree)
    dlat = 0.2  # meridional grid spatial step (in degree)
    dt = np.timedelta64(1, "D")  # temporal grid step
    
    def init_from_config(self, config):
        self.lon_min = config.lon_min
        self.lon_max = config.lon_max
        self.lat_min = config.lat_min
        self.lat_max = config.lat_max
        self.time_min = np.datetime64(config.time_min)
        self.time_max = np.datetime64(config.time_max)
        self.dlon = config.dlon
        self.dlat = config.dlat
        self.dt = np.timedelta64(config.dt_step, config.dt_period) 
        
        return self
    
class AOIParamsSM(AOIParams):
    time_max = np.datetime64("2017-01-31")
    
    
@dataclass
class OIParams:
    Lx = 1.0  # Zonal decorrelation scale (in degree)
    Ly = 1.0  # Meridional decorrelation scale (in degree)
    Lt = 7.0  # Temporal decorrelation scale (in days)
    noise = 0.05

    
def subset_data(ds_obs, aoi_params, oi_params=None):
    
    
    if oi_params is not None:
        
        ds_obs = ds_obs.sel(
            time=slice(
                aoi_params.time_min - np.timedelta64(int(2 * oi_params.Lt), "D"),
                aoi_params.time_max + np.timedelta64(int(2 * oi_params.Lt), "D"),
            ),
            drop=True,
        )
    else:
        
        ds_obs = ds_obs.sel(
            time=slice(
                aoi_params.time_min - np.timedelta64(2, "D"),
                aoi_params.time_max + np.timedelta64(2, "D"),
            ),
            drop=True,
        )
    
    # correct
    ds_obs = correct_lon(ds_obs, aoi_params.lon_min)
    
    # subset lon/lat
    if oi_params is not None:
        ds_obs = ds_obs.where(
            (ds_obs["longitude"] >= aoi_params.lon_min - oi_params.Lx)
            & (ds_obs["longitude"] <= aoi_params.lon_max + oi_params.Lx)
            & (ds_obs["latitude"] >= aoi_params.lat_min - oi_params.Ly)
            & (ds_obs["latitude"] <= aoi_params.lat_max + oi_params.Ly),
            drop=True,
        )
    else:
        ds_obs = ds_obs.where(
            (ds_obs["longitude"] >= aoi_params.lon_min )
            & (ds_obs["longitude"] <= aoi_params.lon_max)
            & (ds_obs["latitude"] >= aoi_params.lat_min)
            & (ds_obs["latitude"] <= aoi_params.lat_max),
            drop=True,
        ) 
    
    # # add a vectorized time
    # ds_obs = add_vtime(ds_obs, aoi_params.time_min)

    # drop all nans
    ds_obs = ds_obs.dropna(dim="time")

    
    return ds_obs