from dataclasses import dataclass
from kernellib.models.exact import init_exact_gp
from kernellib.models.sparse import init_sparse_gp
from kernellib.models.grid import init_kiss_gp


@dataclass
class GPParams:
    model: str = "exact"
    mean_fn: str = "constant"
    scale: str = True
    n_batches_pred: float = 100
    # temporal kernel
    likelihood: str = "gaussian"
    noise: float = 0.05
    
    
    def init_from_config(self, config):
        self.model = config.model
        self.mean_fn = config.mean_fn
        self.scale = config.scale
        self.n_batches_pred = config.n_batches_pred
        self.likelihood = config.likelihood.likelihood
        self.noise = config.likelihood.noise
        
        return self
    

def gp_model_factory(gp_params, kernel_params):
    
    if gp_params.model == "exact":
        return init_exact_gp(gp_params, kernel_params)
    elif gp_params.model == "sparse":
        return init_sparse_gp(gp_params, kernel_params)
    elif gp_params.model == "grid":
        return init_kiss_gp(gp_params, kernel_params)
    
    else:
        raise ValueError(f"Unrecognized kernel option: {kernel}")
        