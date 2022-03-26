from dataclasses import dataclass
import xarray as xr
import numpy as np

@dataclass
class OIParams:
    Lx = 1.0  # Zonal decorrelation scale (in degree)
    Ly = 1.0  # Meridional decorrelation scale (in degree)
    Lt = 7.0  # Temporal decorrelation scale (in days)
    noise = 0.05
    
    def init_from_config(self, config):
        self.Lx = config.Lx
        self.Ly = config.Ly
        self.Lt = config.Lt
        self.noise = config.noise
        
        return self



def oi_params_factory(oi_params: str):
    
    if oi_params == "default":
        return OIParams()
    else:
        raise ValueError(f"Unrecognized oi params: {oi_params}")

        
def create_oi_grid(aoi_params, n_samples=10):
    """ """

    # create grids
    glon = np.arange(aoi_params.lon_min, aoi_params.lon_max + aoi_params.dlon, aoi_params.dlon)
    glat = np.arange(
        aoi_params.lat_min, aoi_params.lat_max + aoi_params.dlat, aoi_params.dlat
    )  # output OI latitude grid
    gtime = np.arange(
        aoi_params.time_min, aoi_params.time_max + aoi_params.dt, aoi_params.dt
    )  # output OI time grid

    nx = len(glon)
    ny = len(glat)
    nt = len(gtime)

    # define & initialize ssh array
    gssh_mu = np.zeros((nt, ny, nx))
    gssh_var = np.zeros((nt, ny, nx))
    gssh_samples = np.zeros((n_samples, nt, ny, nx))
    nobs = np.zeros(nt)

    # Make 2D grid
    glon2, glat2 = np.meshgrid(glon, glat)
    fglon = glon2.flatten()
    fglat = glat2.flatten()

    ng = len(fglat)  # number of grid points
    vtime = (gtime - gtime[0]) / np.timedelta64(1, "D")

    ds_oi_grid = xr.Dataset(
        {
            "gssh_mu": (("time", "lat", "lon"), gssh_mu),
            "gssh_var": (("time", "lat", "lon"), gssh_var),
            "gssh_samples": (("samples", "time", "lat", "lon"), gssh_samples),
            "glon2": (("lat", "lon"), glon2),
            "glat2": (("lat", "lon"), glat2),
            "fglon": (("ng"), fglon),
            "fglat": (("ng"), fglat),
            "nobs": (("time"), nobs),
        },
        coords={
            "gtime": (vtime).astype(np.float),
            "time": gtime,
            "lat": glat,
            "lon": glon,
            "samples": np.arange(n_samples),
            "ng": np.arange(ng),
        },
    )

    return ds_oi_grid