from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior

def mean_fn_factory(mean_fn: str="constant"):
    
    if mean_fn == "constant":
        return ConstantMean()
    elif mean_fn == "constant_prior":
        return ConstantMean(prior=SmoothedBoxPrior(-1e-5, 1e-5))
    else:
        raise ValueError(f"Unrecognized kernel option: {kernel}")