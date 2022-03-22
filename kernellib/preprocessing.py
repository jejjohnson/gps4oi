import xarray as xr
import numpy as np


def create_oi_grid(aoi, n_samples: int = 10):
    """ """

    # create grids
    glon = np.arange(aoi.lon_min, aoi.lon_max + aoi.dlon, aoi.dlon)
    glat = np.arange(
        aoi.lat_min, aoi.lat_max + aoi.dlat, aoi.dlat
    )  # output OI latitude grid
    gtime = np.arange(
        aoi.time_min, aoi.time_max + aoi.dt, aoi.dt
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


def correct_lon(ds, lon_min):

    if lon_min < 0:
        ds["longitude"] = xr.where(
            ds["longitude"] >= 180.0, ds["longitude"] - 360.0, ds["longitude"]
        )
    return ds


def add_vtime(ds, time_min: np.datetime64):
    vtime = (ds["time"].values - time_min) / np.timedelta64(1, "D")
    return ds.assign_coords({"time": vtime})
