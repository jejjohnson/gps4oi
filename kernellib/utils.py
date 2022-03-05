from einops import repeat
from scipy.linalg import cho_factor, cho_solve


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
