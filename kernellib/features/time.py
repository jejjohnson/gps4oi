import numpy as np


def add_vtime(ds, time_min: np.datetime64):
    vtime = (ds["time"].values - time_min) / np.timedelta64(1, "D")
    return ds.assign_coords({"time": vtime})
