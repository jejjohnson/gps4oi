import jax.numpy as jnp


def get_meshgrid(res: float, nx: int, ny: int):
    dx = res
    dy = res
    x = jnp.linspace(-1, 1, int(nx)) * (nx - 1) * dx / 2
    y = jnp.linspace(-1, 1, int(ny)) * (ny - 1) * dy / 2
    return jnp.meshgrid(x, y)


def calculate_gradient(da, edge_order=2):
    
    # first marginal derivative
    dx = da.differentiate(coord="Nx", edge_order=2)
    dy = da.differentiate(coord="Ny", edge_order=2)

    return 0.5 * (dx**2 + dy**2)

def calculate_laplacian(da, edge_order=2):
    
    # second marginal derivative
    dx2 = da.differentiate(coord="Nx", edge_order=2).differentiate(coord="Nx", edge_order=2)
    dy2 = da.differentiate(coord="Ny", edge_order=2).differentiate(coord="Ny", edge_order=2)

    return 0.5 * (dx2**2 + dy2**2)