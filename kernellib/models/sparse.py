import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel, RFFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import SmoothedBoxPrior
from kernellib.models.means import mean_fn_factory



def init_sparse_gp(gp_params, kernel):
    
    # initialize mean function
    mean = mean_fn_factory(gp_params.mean_fn)
    
    
    # initialize likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    n_inducing_points = gp_params.n_inducing
    
    
    class SparseGPRModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y):
            super(SparseGPRModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = mean
            self.base_covar_module = kernel
            self.covar_module = InducingPointKernel(
                self.base_covar_module, 
                inducing_points=train_x[:n_inducing_points, :], 
                likelihood=likelihood
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return SparseGPRModel, likelihood