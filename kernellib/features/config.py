import numpy as np


def _convert_to_time(dt_step, dt_period):
    return np.timedelta64(dt_step, dt_period)


def create_spatial_coords(spatial_min, spatial_max, dx_spatial):
    gspatial = np.arange(
        spatial_min,
        spatial_max + dx_spatial,
        dx_spatial,
    )


    return gspatial

def create_temporal_coords(time_min, time_max, dt_step, dt_period):

    time_min = np.datetime64(time_min)
    time_max = np.datetime64(time_max)
    dt = _convert_to_time(dt_step, dt_period)

    return np.arange(
        time_min,
        time_max + dt,
        dt,
    )

def create_spatiotemporal_coords(lon_min, lon_max, dx_lon, lat_min, lat_max, dx_lat, time_min, time_max, dt_step, dt_period):
    glon = create_spatial_coords(lon_min, lon_max, dx_lon)
    glat = create_spatial_coords(lat_min, lat_max, dx_lat)
    gtime = create_temporal_coords(time_min, time_max, dt_step, dt_period)
    return glon, glat, gtime


