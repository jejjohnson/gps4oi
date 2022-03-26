

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
