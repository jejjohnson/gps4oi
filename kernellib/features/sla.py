import xarray as xr
import numpy as np



def create_sla_grid(glon, glat, gtime):
    """ """

    nx = len(glon)
    ny = len(glat)
    nt = len(gtime)

    # define & initialize ssh array
    gssh_mu = np.empty((nt, ny, nx))
    gssh_var = np.empty((nt, ny, nx))
    nobs = np.empty(nt)

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
            "ng": np.arange(ng),
        },
    )

    return ds_oi_grid