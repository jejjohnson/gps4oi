from einops import repeat
from scipy.linalg import cho_factor, cho_solve
from typing import NamedTuple

def add_dim(x):
    if x.ndim < 2:
        x = x.reshape(-1, 1)
    return x


def broadcast_timesteps(t, n_steps):
    t = repeat(t, "1 ... -> N ...", N=n_steps)
    return t


def cholesky_solve(K, y):
    L = cho_factor(K, lower=True)
    return cho_solve(L, y), L

class ResultsFileName(NamedTuple):
    filename: str
    
    @property
    def stats_filename(self):
        return f"stats_{self.filename}"
    
    @property
    def stats_ts_filename(self):
        return f"stats_ts_{self.filename}"
    
    @property
    def psd_filename(self):
        return f"psd_{self.filename}"
