import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import GridInterpolationKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import SmoothedBoxPrior
from kernellib.models.means import mean_fn_factory



def init_kiss_gp(gp_params, kernel):
    
    # initialize mean function
    mean = mean_fn_factory(gp_params.mean_fn)
    
    
    # initialize likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    
    
    
    class KISSGPRModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y):
            super(KISSGPRModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = mean
            self.base_covar_module = kernel
            grid_size = gpytorch.utils.grid.choose_grid_size(train_x,1.0)
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                self.base_covar_module, grid_size=grid_size, num_dims=3
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return KISSGPRModel, likelihood