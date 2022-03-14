from typing import NamedTuple
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

    


class AOI(NamedTuple):
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    time_min: np.datetime64
    time_max: np.datetime64
    

class SpectralStats(NamedTuple):
    delta_t: float = 0.9434 # seconds
    velocity: float = 6.77 # km/sec
    length_scale: float = 1_000 # sehment length scale in km
    
    @property
    def delta_x(self):
        return self.velocity * self.delta_t
    
    
from typing import NamedTuple

class RMSEBinning(NamedTuple):
    bin_lat_step: float = 1.0
    bin_lon_step: float = 1.0
    bin_time_step: str = "1D"
    