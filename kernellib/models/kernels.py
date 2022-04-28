from dataclasses import dataclass
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel, RFFKernel, MaternKernel

@dataclass
class KernelParams:
    kernel_fn: str = "matern52"
    # spatial kernel
    lon_kernel: str = "matern52"
    lat_kernel: str = "matern52"
    lon_lengthscale: float = 1.0
    lat_lengthscale: float = 1.0
    # temporal kernel
    temporal_kernel: str = "matern52"
    temporal_lengthscale: float = 7.0
    
    
    def init_from_config(self, config):
        self.kernel_fn = config.kernel_fn
        self.lon_kernel = config.lon_kernel
        self.lat_kernel = config.lat_kernel
        self.temporal_kernel = config.temporal_kernel
        self.lat_lengthscale = config.lat_lengthscale
        self.lon_lengthscale = config.lon_lengthscale
        self.temporal_lengthscale = config.temporal_lengthscale
        
        return self
    




def kernel_fn_factory(kernel_fn, params):
    
    if kernel_fn == "rbf":
        # all kernels the same
        kernel = RBFKernel(ard_num_dims=3)
        kernel.lengthscale = [
            params.temporal_lengthscale, 
            params.lat_lengthscale,
            params.lon_lengthscale
        ]
    
        return ScaleKernel(kernel)
    
    elif kernel_fn == "matern12":
        # all kernels the same
        kernel = MaternKernel(nu=0.5, ard_num_dims=3)
        kernel.lengthscale = [
            params.temporal_lengthscale, 
            params.lat_lengthscale,
            params.lon_lengthscale
        ]
    
        return ScaleKernel(kernel)
    
    elif kernel_fn == "matern32":
        kernel = MaternKernel(nu=1.5, ard_num_dims=3)
        kernel.lengthscale = [
            params.temporal_lengthscale, 
            params.lat_lengthscale,
            params.lon_lengthscale
        ]
    
        return ScaleKernel(kernel)
    
    elif kernel_fn == "matern52":
        kernel = MaternKernel(nu=2.5, ard_num_dims=3)
        kernel.lengthscale = [
            params.temporal_lengthscale, 
            params.lat_lengthscale,
            params.lon_lengthscale
        ]
    
        return ScaleKernel(kernel)
    
    elif kernel_fn == "mixture":
        return NotImplementedError()
    
    elif kernel_fn == "spatiotemp":
        return NotImplementedError()
    
    elif kernel_fn == "prod":
        return NotImplementedError()
    
    elif kernel_fn == "rff":
        return NotImplementedError()
    
    else:
        raise ValueError(f"Unrecognized kernel option: {kernel}")

def get_rbf_kernel():
    
    return ScaleKernel(RBFKernel(ard_num_dims=None))

def get_ard_kernel(ard_num_dims: int):
    
    return ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))


def get_inducing_point_kernel(base_cov, likelihood, z):
    
    return InducingPointKernel(
                    base_cov,
                    inducing_points=z,
                    likelihood=likelihood,
                )