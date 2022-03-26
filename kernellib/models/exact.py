import gpytorch
from dataclasses import dataclass, field
from typing import Optional, List
from gpytorch.distributions import MultivariateNormal
from kernellib.models.means import mean_fn_factory
from kernellib.models.kernels import kernel_fn_factory


def init_exact_gp(gp_params, kernel):
    
    # initialize mean function
    mean = mean_fn_factory(gp_params.mean_fn)
    
    
    # initialize likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = mean
            self.covar_module = kernel

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return GPRegressionModel, likelihood


def init_exact_spatiotemp_gp(gp_params):
    
    # initialize mean function
    mean = mean_fn_factory(gp_params.mean_fn)
    
    # initialize temporal kernel
    temporal_kernel = kernel_fn_factory(gp_params.kernel_fn, ard_num_dims=1, active_dims=[0])
    temporal_kernel.lengthscale = gp_params.length_scale[0]
    
    # initialize spatial kernel
    spatio_kernel = kernel_fn_factory(gp_params.kernel_fn, ard_num_dims=2, active_dims=[1,2])
    spatio_kernel.lengthscale = gp_params.length_scale[1:]
    
    kernel = temporal_kernel * spatio_kernel
    
    print(kernel)
    
    # initialize likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = mean
            self.covar_module = kernel

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            
            return MultivariateNormal(mean_x, covar_x)

    return GPRegressionModel, likelihood



def init_exact_sum_gp(gp_params):
    
    # initialize mean function
    mean = mean_fn_factory(gp_params.mean_fn)
    
    # initialize temporal kernel
    temporal_kernel = kernel_fn_factory(gp_params.kernel_fn, ard_num_dims=1, active_dims=[0])
    temporal_kernel.lengthscale = gp_params.length_scale[0]
    
    # initialize spatial kernel
    lat_kernel = kernel_fn_factory(gp_params.kernel_fn, ard_num_dims=1, active_dims=[1])
    lat_kernel.lengthscale = gp_params.length_scale[1]
    
    # initialize spatial kernel
    lon_kernel = kernel_fn_factory(gp_params.kernel_fn, ard_num_dims=1, active_dims=[2])
    lon_kernel.lengthscale = gp_params.length_scale[2]
    
    kernel = temporal_kernel + lat_kernel + lon_kernel
    
    # initialize likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = mean
            self.covar_module = kernel

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            
            return MultivariateNormal(mean_x, covar_x)
    return GPRegressionModel, likelihood

@dataclass
class ExactGPModel:
    mean_fn: str = "constant"
    kernel_fn: str = "rbf"
    ard_dims: int = 3
    length_scale: Optional[List[float]] = field(default_factory=list)
    scale: bool = True
    likelihood: str = "gaussian"
    noise: int = 0.05
    model: str = "exact"
    