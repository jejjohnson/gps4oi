import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel, RFFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import SmoothedBoxPrior


def get_exact_gp():
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, ard_num_dims: int=3):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return GPRegressionModel


def get_sparse_gp():
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, n_inducing_points: int = 500, ard_num_dims: int=3):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.base_covar_module = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))
            self.covar_module = InducingPointKernel(
                self.base_covar_module,
                inducing_points=train_x[:n_inducing_points, :],
                likelihood=likelihood,
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return GPRegressionModel


def get_rff_gp():
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, num_samples: int = 100, ard_num_dims: int=3):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1e-5, 1e-5))
            self.covar_module = ScaleKernel(
                RFFKernel(num_samples=num_samples, ard_num_dims=ard_num_dims)
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return GPRegressionModel
