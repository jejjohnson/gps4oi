from dataclasses import dataclass
from einops import repeat
import numpy as np
from kernellib.utils import add_dim


@dataclass
class Dimensions:
    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray

    def coord_vector(self):
        lat = self.lat
        lon = self.lon
        time = np.atleast_1d(self.time)

        if time.shape[0] == 1:
            time = repeat(time, "1 -> N", N=lat.shape[0])

        vec = np.vstack([time, lat, lon]).T
        assert vec.shape[1] == 3
        assert vec.shape[0] == lat.shape[0] == lon.shape[0]
        return vec


@dataclass
class GeoData(Dimensions):
    data: np.ndarray
